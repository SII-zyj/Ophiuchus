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

from typing import List
import re


def _extract_tool_or_answer(text: str) -> str:
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    tool_match = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.IGNORECASE | re.DOTALL)
    if tool_match:
        return tool_match.group(1).strip()

    action_match = re.search(r"action\s*[:：]\s*([A-Za-z0-9_\- ]+)", text, flags=re.IGNORECASE)
    if action_match:
        return action_match.group(1).strip()

    tag_match = re.search(r"<action>(.*?)</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()

    return text[-50:]


def medical_agent_projection(actions: List[str]):
    """Process LLM outputs to extract structured actions for the environment."""

    valids = [0] * len(actions)
    processed_actions: List[str] = [""] * len(actions)

    for i, action in enumerate(actions):
        original_str = action or ""
        action_lower = original_str.lower()

        extracted_action = _extract_tool_or_answer(original_str)
        cleaned_action = original_str.strip()
        processed_actions[i] = cleaned_action if cleaned_action else extracted_action

        think_start_idx = action_lower.find("<think>")
        think_end_idx = action_lower.find("</think>")
        if think_start_idx != -1 and think_end_idx != -1:
            valids[i] = 1

        if re.search(r"[\u4e00-\u9fff]", original_str):
            valids[i] = 0

    return processed_actions, valids