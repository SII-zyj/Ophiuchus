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
# FILE: agent_system/environments/prompts/medical_agent.py
#
# PURPOSE:
#   Defines all prompt strings injected into the model's conversation at each
#   stage of a medical-agent episode. These are plain string constants — they
#   contain no logic, only text that instructs the model how to behave.
#
# 文件作用：
#   定义医学智能体每轮对话中注入给模型的所有提示词字符串。
#   这里只有纯文本常量，不包含任何逻辑。
# =============================================================================


# -----------------------------------------------------------------------------
# TOOL DESCRIPTION — injected inside the system prompt.
# Tells the model what three tools exist, their purpose, and their JSON input
# format so the model can construct valid tool calls.
#
# 工具说明 — 插入系统提示词中。
# 告诉模型有哪三个工具、各自的用途和 JSON 输入格式，
# 使模型能够构造合法的工具调用。
# -----------------------------------------------------------------------------
MEDICAL_AGENT_TOOL_DESCRIPTION = """### Available tools:
You can use the following three tools to process the images. After each tool usage, you must wait for and analyze the visualization feedback before proceeding.

1. **Zoom-in**
- Purpose: Zoom in on a specific region of an image by cropping it to a bounding box for detailed inspection. If a mask is provided, the zoomed image will highlight the mask's contour.
- Input format: JSON
```json
[
    {
        "index": i, # Image index
        "bbox_2d": [x1, y1, x2, y2]
    }
]
```
- Output: Generates zoomed areas for visual inspection of the i-th image

2. **BiomedParse**
- Purpose: Detect and segment a specified object type in the image (e.g. lesion, tumor) using text descriptions for the targets.
- Input format: JSON
```json
[
    {
        "index": i, # Image index
        "captions": "target_description"
    }
]
```
- Output: Generates segmentation masks for target objects of the i-th image

3. **SAM2**
- Purpose: Detect and Segment an object in the image given a bounding box.
- Input format: JSON
```json
[
    {
        "index": i, # Image index
        "bbox_2d": [x1, y1, x2, y2]
    }
]
```
- Output: Generates segmentation masks for target objects of the i-th image
"""

# -----------------------------------------------------------------------------
# REQUIRED FORMAT — injected inside the system prompt.
# Enforces two output formats the model must follow on every turn:
#   (a) Tool call turn:  <think>...</think>  Action: <ToolName>  ```json...```
#   (b) Final answer:    <think>...</think>  Action: Answer  <answer>...</answer>
# The reward function checks this format and gives a format reward of +1.
#
# 输出格式要求 — 插入系统提示词中。
# 强制要求模型在每轮输出两种格式之一：
#   (a) 工具调用轮：<think>...</think> Action: <工具名> ```json...```
#   (b) 最终回答：  <think>...</think> Action: Answer <answer>...</answer>
# 奖励函数会检查此格式，格式正确则给 +1 的格式奖励。
# -----------------------------------------------------------------------------
MEDICAL_AGENT_REQUIRED_FORMAT = """### Required Output Format:
For each reasoning step, you must structure your response as follows:
<think> [Your detailed reasoning process] </think> Action: [Zoom-in/BiomedParse/SAM2]
```json
[JSON format coordinates or descriptions]
```

After your reasoning and iteratively refine your solution through tool invocation feedback, you should arrive at a final answer and structure your response as follows:
<think> [Your detailed reasoning process] </think> Action: Answer
<answer> [Your final answer] </answer>
"""

# -----------------------------------------------------------------------------
# REASONING TIPS — injected inside the system prompt.
# Guides the model to perform initial analysis, choose tools deliberately,
# evaluate tool outputs critically, and backtrack when needed.
# This shapes the thinking style during RL training.
#
# 推理技巧提示 — 插入系统提示词中。
# 引导模型做初始分析、有目的地选择工具、批判性地评估工具输出、
# 必要时回溯修正。这会影响 RL 训练中模型形成的推理风格。
# -----------------------------------------------------------------------------
MEDICAL_AGENT_REASONING_TIPS = """### Please NOTE the following reasoning techniques:
1. Initial Analysis
   - Break down the complex problem
   - Plan your approach

2. Iterative Reasoning for Each Step
   - Choose appropriate tool
   - Provide absolute coordinates in JSON format (The top-left corner of the image is (0, 0) and the bottom-right corner is (512, 512))
   - Observe the tool invocation output
   - Reflect on the results returned by the tool:
     * Does the results of the segmentation or zooming reasonable?
     * Does it align with your reasoning?
     * What adjustments are needed?
   - Backtrack and Adjust:
     * If errors found, backtrack to previous step to modify actions or decisions as needed.
"""

# -----------------------------------------------------------------------------
# SYSTEM PROMPT — sent as message[0] with role="system" at episode start.
# Assembled from the three blocks above. This is the first thing the model
# reads when a new case begins. It never changes within an episode.
#
# 系统提示词 — 在 episode 开始时作为 message[0]（role="system"）发送。
# 由上面三个模块拼接而成。这是模型在新案例开始时读到的第一条消息，
# 在整个 episode 内保持不变。
# -----------------------------------------------------------------------------
MEDICAL_AGENT_SYSTEM_PROMPT = f"""### Guidance:
You are a helpful assistant specialized in medical image analysis. You have access to several tools that help you segment and examine medical images (e.g. highlighting lesions or tumors) to answer questions.
Your task is to carefully analyze the image and question, use the tools step-by-step, and provide a well-reasoned final answer through tool invocation feedback.

{MEDICAL_AGENT_TOOL_DESCRIPTION}

{MEDICAL_AGENT_REQUIRED_FORMAT}

{MEDICAL_AGENT_REASONING_TIPS}
"""

# -----------------------------------------------------------------------------
# USER PROMPT — sent as message[1] with role="user" at episode start.
# Contains: the <image> placeholder (replaced by image tokens during encoding),
# the actual clinical question, the MCQ options, and the image metadata
# (index, width, height) so the model can reference the image correctly in
# its tool-call JSON.
#
# 用户提示词 — 在 episode 开始时作为 message[1]（role="user"）发送。
# 包含：<image> 占位符（编码时替换为图像 token）、具体临床问题、
# MCQ 选项、以及图像元信息（编号、宽、高），
# 使模型在工具调用 JSON 中能正确引用图像。
# -----------------------------------------------------------------------------
MEDICAL_AGENT_USER_PROMPT = """<image>
### Question:
{question}
Options:
{options}
The index of the given image is {image_index} (width: {width}, height: {height}).
Begin your reasoning. After each tool use, critically evaluate the image returned by the tool and adjust tool decisions if needed:
"""

# -----------------------------------------------------------------------------
# TOOL FEEDBACK — sent as a new user message after each tool call completes.
# Contains a new <image> placeholder for the tool's output image (mask overlay
# or zoom crop), plus the new image index and dimensions. This closes the
# observation loop: model acts → environment calls tool → environment replies
# with this message + the output image → model reads and continues reasoning.
#
# 工具反馈提示词 — 每次工具调用完成后作为新的 user 消息发送。
# 包含工具输出图像的 <image> 占位符（mask 叠加图或裁剪图）、
# 新图像的编号和尺寸。
# 这形成了观察闭环：模型输出动作 → 环境调用工具 → 环境用此消息
# + 输出图像回复给模型 → 模型读取后继续推理。
# -----------------------------------------------------------------------------
MEDICAL_AGENT_TOOL_FEEDBACK = """<image>
The index of the given image is {image_index} (width: {width}, height: {height}). Continue your reasoning. After each tool use, critically evaluate the image returned by the tool and adjust tool decisions if needed:
"""
