# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# FILE: agent_system/reward_manager/episode.py
#
# PURPOSE:
#   Converts the scalar episode reward produced by the environment into a
#   reward *tensor* that the GRPO / PPO algorithm can consume. This is the
#   bridge between the environment world (scalar float) and the RL training
#   world (dense tensor aligned with token positions).
#
#   The key design choice: reward is SPARSE — only the very last valid
#   response token in each trajectory receives a non-zero reward. All other
#   token positions stay at 0. This is standard practice for episode-level
#   (sparse) RL on language models.
#
#   Called by: verl/trainer/ppo/ray_trainer.py after each rollout batch.
#   Input:     DataProto containing full trajectory (prompts + responses +
#              episode_rewards stored in non_tensor_batch).
#   Output:    reward_tensor of shape (batch, response_length), with the
#              episode reward placed at position [i, valid_response_length-1].
#
# 文件作用：
#   将环境产生的标量 episode 奖励转换为 GRPO/PPO 算法可消费的奖励张量。
#   这是环境世界（标量浮点）与 RL 训练世界（与 token 位置对齐的稠密张量）
#   之间的桥梁。
#
#   关键设计：奖励是稀疏的 —— 只有每条轨迹最后一个有效响应 token 处
#   才有非零奖励，其余 token 位置均为 0。
#   这是语言模型 episode 级（稀疏）RL 的标准做法。
#
#   调用方：verl/trainer/ppo/ray_trainer.py，在每批 rollout 后调用。
#   输入：DataProto，包含完整轨迹（prompt + response + episode_rewards）。
#   输出：shape 为 (batch, response_length) 的 reward_tensor，
#         episode 奖励放置在 [i, valid_response_length-1] 处。
# =============================================================================

from verl import DataProto
import torch
import numpy as np


class EpisodeRewardManager:
    """
    Reward manager for episode-level (sparse) reward signals.

    This class is registered as the reward function in the PPO trainer config.
    It does NOT compute rewards — those come from the environment. It only
    reshapes the scalar episode_reward into the token-aligned tensor format
    that the actor's loss function expects.

    用于 episode 级（稀疏）奖励信号的奖励管理器。
    该类在 PPO 训练器配置中注册为奖励函数。
    它不计算奖励（奖励来自环境），只是将标量 episode_reward
    重塑为 actor 损失函数所需的 token 对齐张量格式。
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        """
        Args:
            tokenizer:           Used only for decoding and printing sample
                                 trajectories to the console for inspection.
            num_examine:         How many unique data sources to print decoded
                                 examples for (for debugging / monitoring).
            normalize_by_length: If True, divide the episode reward by the
                                 number of steps taken, rewarding efficiency.
                                 Default False (use raw episode reward).

        tokenizer:           仅用于将 token id 解码为文本并打印到控制台供调试。
        num_examine:         对多少个不同数据源打印解码样本（调试/监控用）。
        normalize_by_length: 若为 True，将 episode 奖励除以步数，
                             鼓励模型更高效地解决问题。默认 False。
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict=False):
        """
        Build the reward tensor from a batch of completed trajectories.

        Args:
            data:        DataProto batch. Each item contains:
                           batch['prompts']           — prompt token ids
                           batch['responses']         — response token ids
                           batch['attention_mask']    — 1 for real tokens, 0 for padding
                           non_tensor_batch['episode_rewards']  — scalar from env
                           non_tensor_batch['episode_lengths']  — number of steps taken
            return_dict: If True, return {'reward_tensor': ..., 'reward_extra_info': {}}.
                         If False, return the tensor directly.

        Returns:
            reward_tensor: FloatTensor of shape (batch_size, response_length).
                           Zero everywhere except position [i, last_valid_token],
                           which holds the episode reward for trajectory i.

        从一批已完成轨迹中构建奖励张量。

        返回：shape 为 (batch_size, response_length) 的 FloatTensor。
              除 [i, last_valid_token] 位置（存放 episode 奖励）外全为零。
        """
        # Short-circuit: if a reward model already scored these responses,
        # use those scores directly without re-computing.
        # 短路：若奖励模型已对这些响应打分，直接使用，跳过后续计算。
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Initialize reward tensor to all zeros (sparse reward baseline).
        # Shape: (batch_size, response_length)
        # 初始化奖励张量为全零（稀疏奖励基准）。
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Track how many examples we've printed per data source (for logging).
        # 跟踪每个数据源已打印的样本数量（用于日志）。
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            # --- Locate the real (non-padded) prompt and response tokens ---
            # --- 定位真实（非填充）的 prompt 和 response token ---
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            # attention_mask covers both prompt and response; split at prompt_length.
            # attention_mask 覆盖 prompt 和 response，在 prompt_length 处分割。
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]   # strip left-padding

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]  # strip right-padding

            # Decode for optional console logging.
            # 解码用于可选的控制台日志输出。
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch['data_source']

            # --- Retrieve the episode-level reward from the environment ---
            # --- 从环境获取 episode 级奖励 ---
            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            # Optionally normalize by episode length to reward efficiency.
            # 可选：除以 episode 步数，以鼓励效率。
            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards

            # Place the scalar reward at the LAST valid response token position.
            # This implements sparse reward: only the final token gets a signal.
            # The PPO loss uses this tensor to weight the policy gradient.
            # 将标量奖励放在最后一个有效响应 token 的位置。
            # 实现稀疏奖励：只有最终 token 获得信号。
            # PPO 损失用此张量对策略梯度加权。
            reward_tensor[i, valid_response_length - 1] = torch.tensor(
                score, dtype=torch.float32, device=prompt_ids.device
            )

            # --- Optional: print a sample trajectory to console for monitoring ---
            # --- 可选：将样本轨迹打印到控制台以便监控 ---
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
