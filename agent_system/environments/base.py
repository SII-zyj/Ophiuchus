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
# FILE: agent_system/environments/base.py
#
# PURPOSE:
#   Defines the abstract base class EnvironmentManagerBase that every
#   concrete environment manager (MedicalEnvironmentManager, etc.) must
#   subclass. It establishes the shared interface that rollout_loop.py
#   expects when it calls envs.reset() and envs.step().
#
#   Think of this as the "contract" file: any new environment you add must
#   implement reset(), step(), and optionally success_evaluator().
#
#   Also provides to_numpy(), a small utility used throughout the codebase
#   to safely convert various types to numpy arrays.
#
# 文件作用：
#   定义抽象基类 EnvironmentManagerBase，所有具体的环境管理器
#   （MedicalEnvironmentManager 等）都必须继承它。
#   该基类建立了 rollout_loop.py 在调用 envs.reset() 和 envs.step() 时
#   所期望的共享接口。
#
#   可以把这个文件理解为"契约"：任何新增的环境都必须实现
#   reset()、step()，以及可选的 success_evaluator()。
#
#   同时提供 to_numpy() 工具函数，广泛用于将各种类型安全转换为 numpy 数组。
# =============================================================================

from typing import List, Tuple, Dict, Union, Any
import torch
import numpy as np
import os
from agent_system.environments.prompts import *
from collections import defaultdict


def to_numpy(data):
    """
    Safely convert a variety of types to a numpy array.
    Handles: torch.Tensor (detach+cpu first), np.ndarray (passthrough),
             Python scalars and sequences.
    Raises ValueError for unsupported types.

    将多种类型安全转换为 numpy 数组。
    支持：torch.Tensor（先 detach+cpu）、np.ndarray（直通）、
          Python 标量和序列。不支持的类型抛出 ValueError。
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data


class EnvironmentManagerBase:
    """
    Abstract base class for all environment managers.

    Responsibilities:
      - Wraps a vectorized environment (self.envs) that handles N parallel
        episodes (one per sample in the training batch).
      - Applies self.projection_f to convert raw LLM text output into
        environment-compatible action strings before each step.
      - Returns observations in a standardized dict format:
          {'text': ..., 'image': ..., 'anchor': ...}
      - Evaluates episode success after rollout completes.

    Subclasses override reset(), step(), and build_text_obs() to handle
    environment-specific observation formatting.

    所有环境管理器的抽象基类。

    职责：
      - 包装向量化环境（self.envs），并行处理 N 个 episode。
      - 在每步之前，用 self.projection_f 将原始 LLM 文本输出转换为
        环境兼容的动作字符串。
      - 以标准化字典格式返回观察结果：{'text':..., 'image':..., 'anchor':...}
      - rollout 完成后评估 episode 是否成功。

    子类通过覆写 reset()、step()、build_text_obs() 处理特定环境的观察格式。
    """

    def __init__(self, envs, projection_f, config):
        """
        Args:
            envs:         Vectorized environment (e.g. MedicalAgentEnvs).
                          Manages N Ray remote workers, one per parallel episode.
            projection_f: Callable mapping List[str] (raw LLM outputs) to
                          (processed_actions, valids). Defined in projection.py.
            config:       Full OmegaConf training config from main_ppo.py.

        envs:         向量化环境实例（如 MedicalAgentEnvs），
                      管理 N 个 Ray 远程 worker，每个并行 episode 对应一个。
        projection_f: 将 List[str]（原始 LLM 输出）映射为
                      (processed_actions, valids) 的可调用对象。
        config:       从 main_ppo.py 透传的完整 OmegaConf 训练配置。
        """
        self.envs = envs
        self.projection_f = projection_f
        self.config = config

    def reset(self, kwargs) -> Dict[str, Any]:
        """
        Reset all parallel environments and return initial observations.

        Default: delegates to self.envs.reset(), returns image-only obs.
        Subclasses override to add text formatting, memory init, etc.

        Returns:
            observations: dict with keys 'text', 'image', 'anchor'.
            infos:        list of per-environment info dicts.

        重置所有并行环境并返回初始观察结果。
        默认：委托给 self.envs.reset()，只返回图像观察。
        子类覆写以添加文本格式化、记忆初始化等。
        """
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos

    def step(self, text_actions: List[str]):
        """
        Execute one step across all parallel environments.

        Flow:
          1. projection_f: raw LLM text → clean actions + validity flags.
          2. self.envs.step(actions): advance each environment by one step.
          3. Attach is_action_valid to each info dict for the trainer.
          4. Return standardized observation dict + rewards + dones + infos.

        Subclasses override for environment-specific post-processing.

        在所有并行环境中执行一步。

        流程：
          1. projection_f：原始 LLM 文本 → 清洗动作 + 合法性标志。
          2. self.envs.step(actions)：每个环境前进一步。
          3. 将 is_action_valid 附加到每个 info 字典。
          4. 返回标准化观察字典 + 奖励 + 完成标志 + infos。
        """
        # Parse raw LLM output into clean actions and format-validity flags.
        # 将原始 LLM 输出解析为清洗后的动作和格式合法性标志。
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None,    # Subclasses fill this.  子类填充此项。
            'image': next_obs,
            'anchor': None   # GiGPO only.  仅供 GiGPO 使用。
        }
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def build_text_obs(self,) -> List[str]:
        """
        Format raw text observations into model-ready strings.
        Must be implemented by text-based environment subclasses.

        将原始文本观察格式化为模型可直接使用的字符串。
        基于文本的环境子类必须实现此方法。
        """
        pass

    def close(self) -> None:
        """Release all environment resources (Ray actors, sockets, etc.)."""
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate episode success over an entire rollout batch.
        Called once per training step after all episodes complete.

        Default: checks info['won'] in the last active step of each episode.

        Returns:
            dict mapping metric names → np.ndarray of shape (batch_size,).
            Always includes 'success_rate' (1.0 = won, 0.0 = lost).

        对整个 rollout 批次的 episode 进行成功率评估。
        在所有 episode 完成后每个训练步骤调用一次。
        默认：检查每个 episode 最后一个活跃步骤的 info['won']。

        返回：指标名称 → shape 为 (batch_size,) 的 np.ndarray 的字典，
              始终包含 'success_rate'。
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)

        success = defaultdict(list)

        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)

        assert len(success['success_rate']) == batch_size
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """
        Scan episode batch_idx in reverse to find the last active step,
        then record its info['won'] value. Skips padding steps where
        active_masks=False (environment already done, no new data).

        逆序扫描 episode batch_idx，找到最后一个活跃步骤，
        记录其 info['won'] 值。跳过 active_masks=False 的填充步骤
        （环境已结束，无新数据）。
        """
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def save_image(self, image, step):
        """
        Debug utility: save an intermediate image observation to disk.
        Handles tensor/numpy conversion and CHW→HWC transposition automatically.

        调试工具：将中间图像观察保存到磁盘。
        自动处理 tensor/numpy 转换和 CHW→HWC 维度转置。
        """
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.config.env.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")

        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)

        from PIL import Image
        image = Image.fromarray(image)
        image.save(path)
