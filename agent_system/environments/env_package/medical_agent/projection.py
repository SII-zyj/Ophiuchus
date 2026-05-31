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
# FILE: agent_system/environments/env_package/medical_agent/projection.py
#
# PURPOSE:
#   The "projection" layer sits between the raw LLM text output and the
#   environment's step() function. It does two jobs:
#     1. Extract the meaningful action from the model's full response text.
#     2. Decide whether the response is "valid" (well-formatted), which is
#        used to apply an invalid-action penalty during RL training.
#
#   Called by: env_manager.py → MedicalEnvironmentManager.step()
#   Output flows into: envs.py → MedicalAgentWorker.step()
#
# 文件作用：
#   "投影层"位于原始 LLM 文本输出和环境 step() 函数之间，完成两件事：
#     1. 从模型的完整响应文本中提取有意义的动作。
#     2. 判断响应是否"合法"（格式正确），用于 RL 训练中施加无效动作惩罚。
#
#   调用方：env_manager.py → MedicalEnvironmentManager.step()
#   输出流向：envs.py → MedicalAgentWorker.step()
# =============================================================================

from typing import List
import re


def _extract_tool_or_answer(text: str) -> str:
    """
    Try multiple regex patterns to pull the key action out of model output.
    Returns the first match found, in priority order:
      1. <answer>...</answer>         — model is giving its final answer
      2. <tool_call>...</tool_call>   — alternative tool-call tag style
      3. Action: <name>               — the primary expected format
      4. <action>...</action>         — another alternative tag style
      5. last 50 chars of text        — fallback if nothing matched

    This function is only used as a fallback inside medical_agent_projection
    when the cleaned action string is empty. The environment's own parser
    (_parse_tool_invocations in envs.py) does more precise extraction later.

    从模型输出中按优先顺序依次尝试多种正则，提取关键动作：
      1. <answer>...</answer>         — 最终答案
      2. <tool_call>...</tool_call>   — 备选工具调用标签格式
      3. Action: <name>               — 主要预期格式
      4. <action>...</action>         — 另一种备选标签格式
      5. 文本最后 50 个字符           — 所有匹配失败时的兜底
    """
    # Pattern 1: final answer tag
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # Pattern 2: explicit tool_call tag (alternative format)
    tool_match = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.IGNORECASE | re.DOTALL)
    if tool_match:
        return tool_match.group(1).strip()

    # Pattern 3: "Action: <ToolName>" — the primary format enforced by the prompt
    action_match = re.search(r"action\s*[:：]\s*([A-Za-z0-9_\- ]+)", text, flags=re.IGNORECASE)
    if action_match:
        return action_match.group(1).strip()

    # Pattern 4: <action> tag (another alternative)
    tag_match = re.search(r"<action>(.*?)</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()

    # Fallback: return the tail of the response (last 50 chars)
    return text[-50:]


def medical_agent_projection(actions: List[str]):
    """
    Process a batch of raw LLM response strings into cleaned actions and
    validity flags. Called once per rollout step for the whole batch.

    Args:
        actions: List of raw text responses, one per parallel environment.

    Returns:
        processed_actions: List[str] — cleaned response strings passed to
                           MedicalAgentWorker.step(). The worker then does
                           its own fine-grained parsing (_parse_tool_invocations).
        valids: List[int] — 1 if the response is well-formatted, 0 otherwise.
                           Used by env_manager to set info['is_action_valid'],
                           which triggers an invalid-action penalty in the
                           actor's loss if actor.use_invalid_action_penalty=True.

    Validity rules (both must pass for valid=1):
        PASS:  <think>...</think> is present in the response.
        FAIL:  Response contains Chinese characters (unicode CJK range).
               This catches cases where the model reverts to Chinese, which
               indicates it is not following the required English format.

    对一批原始 LLM 响应字符串进行处理，返回清洗后的动作和合法性标志。
    每个 rollout 步骤对整个批次调用一次。

    合法性规则（两条都需满足才为 valid=1）：
        通过：响应中包含 <think>...</think>
        失败：响应包含中文字符（Unicode CJK 范围）
              这用于捕捉模型退化为中文输出的情况，说明未遵循英文格式要求。
    """
    valids = [0] * len(actions)
    processed_actions: List[str] = [""] * len(actions)

    for i, action in enumerate(actions):
        original_str = action or ""
        action_lower = original_str.lower()

        # Use _extract_tool_or_answer only as fallback when the string is empty.
        # Otherwise pass the full cleaned string to the environment so the
        # environment's own parser can do fine-grained extraction.
        # 仅在字符串为空时才用 _extract_tool_or_answer 作兜底；
        # 否则将完整清洗后的字符串传给环境，让环境自己的解析器做精细提取。
        extracted_action = _extract_tool_or_answer(original_str)
        cleaned_action = original_str.strip()
        processed_actions[i] = cleaned_action if cleaned_action else extracted_action

        # Validity check 1: must contain a <think>...</think> block.
        # 合法性检查1：必须包含 <think>...</think> 块。
        think_start_idx = action_lower.find("<think>")
        think_end_idx = action_lower.find("</think>")
        if think_start_idx != -1 and think_end_idx != -1:
            valids[i] = 1

        # Validity check 2: must not contain Chinese characters.
        # Unicode range 一-鿿 covers the CJK Unified Ideographs block.
        # 合法性检查2：不得包含中文字符。
        # 一-鿿 覆盖 CJK 统一汉字区。
        if re.search(r"[一-鿿]", original_str):
            valids[i] = 0

    return processed_actions, valids
