<div align="center">

<img src="asset/logo.png" width="72" alt="Ophiuchus logo" />

# Ophiuchus

### Incentivizing Tool-augmented Thinking with Images for Medical Image Analysis

<p align="center">
  <b>ICML 2026 Accepted</b> · Tool-augmented medical MLLM · Think with images · Segmentation and zoom-in grounded reasoning
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.14157"><img src="https://img.shields.io/badge/arXiv-2512.14157-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2512.14157"><img src="https://img.shields.io/badge/PDF-Paper-0B1F3A?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Paper PDF"></a>
  <a href="https://icml.cc/"><img src="https://img.shields.io/badge/ICML-2026_Accepted-4C6FFF?style=for-the-badge" alt="ICML 2026 Accepted"></a>
  <a href="https://github.com/SII-zyj/Ophiuchus"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/papers/2512.14157"><img src="https://img.shields.io/badge/HF-Paper-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000" alt="Hugging Face Paper"></a>
</p>

<p align="center">
  <a href="https://github.com/SII-zyj/Ophiuchus/stargazers"><img src="https://img.shields.io/github/stars/SII-zyj/Ophiuchus?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/SII-zyj/Ophiuchus/network/members"><img src="https://img.shields.io/github/forks/SII-zyj/Ophiuchus?style=social" alt="GitHub forks"></a>
  <a href="https://doi.org/10.48550/arXiv.2512.14157"><img src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2512.14157-blue" alt="DOI"></a>
  <img src="https://img.shields.io/badge/Models-coming_soon-lightgrey" alt="Models coming soon">
  <img src="https://img.shields.io/badge/Dataset-coming_soon-lightgrey" alt="Dataset coming soon">
</p>

<h4 align="center">
If you find this project useful, please give us a star.
</h4>

</div>

---

## 🎉 News

- **[2026-05-15]** Ophiuchus has been accepted by **ICML 2026**. Congratulations to all authors and contributors.
- **[2025-12-16]** The paper is available on arXiv: [Incentivizing Tool-augmented Thinking with Images for Medical Image Analysis](https://arxiv.org/abs/2512.14157).
- **Coming soon** Model checkpoints, dataset release, and additional evaluation details will be updated in this repository.

---

## 📌 Table of Contents

- [Introduction](#-introduction)
- [Why Ophiuchus](#-why-ophiuchus)
- [Key Highlights](#-key-highlights)
- [Framework Overview](#-framework-overview)
- [Toolbox](#-toolbox)
- [Case Showcase](#-case-showcase)
- [Results](#-results)
- [Citation and Growth](#-citation-and-growth)
- [Data Preparation](#-data-preparation)
- [Training](#️-training)
- [Model](#-model)
- [TODO](#-todo)
- [Acknowledgement](#-acknowledgement)
- [Citation](#-citation)

---

## ⚡ Introduction

<div align="center">
  <img src="asset/intro.png" width="850" alt="Ophiuchus introduction" />
</div>

**Ophiuchus** is a tool-augmented medical multimodal large language model framework that learns to **think with images**. Instead of relying on a single global view, Ophiuchus performs iterative visual evidence acquisition during reasoning.

It learns to decide:

- **When** additional visual evidence is needed.
- **Where** to zoom, segment, and inspect fine-grained structures.
- **How** to integrate localized visual observations into interleaved multimodal reasoning.

The core loop is simple and clinically motivated:

<div align="center">

**Thought → Tool Call → Observation → Grounded Reasoning → Final Answer**

</div>

This allows Ophiuchus to inspect suspicious regions, verify uncertain hypotheses, and reduce hallucinations by grounding its answers in retrieved image evidence.

---

## 🧭 Why Ophiuchus

Medical image analysis often requires local evidence. Subtle lesions, boundaries, textures, and anatomical relationships may be missed by a single global image encoding. Ophiuchus addresses this limitation with an agentic visual reasoning process.

| Challenge | Ophiuchus response |
| --- | --- |
| Global views miss subtle findings | Targeted zoom-in retrieves fine-grained evidence |
| Bounding boxes can be unreliable | Text-driven segmentation delegates localization to external tools |
| Tool use may be unnecessary for easy cases | The model learns when to answer directly and when to inspect further |
| Visual hallucination can derail reasoning | Self-reflection and tool observations help the model correct itself |
| Medical tasks are diverse | The same framework supports VQA, detection, understanding, and reasoning-based segmentation |

---

## 🔥 Key Highlights

- **ICML 2026 accepted** medical AI framework for tool-augmented visual reasoning.
- **Three-tool agentic toolbox** with SAM2, BiomedParse, and zoom-in operations.
- **64K Tool-Integrated VQA** trajectories with explicit tool use and visual grounding.
- **Three-stage training recipe**:
  1. Cold-start supervised fine-tuning.
  2. Self-reflection fine-tuning.
  3. Agentic Tool Reinforcement Learning, ATRL.
- **Strong zero-shot generalization** across medical VQA benchmarks.
- **Joint medical segmentation, understanding, and reasoning** within one agentic framework.
- **Training-free tool transfer** by replacing the segmentation tool while preserving the learned tool-use policy.

---

## 🧩 Framework Overview

<div align="center">
  <img src="asset/method.png" width="1100" alt="Ophiuchus method" />
</div>

Ophiuchus is trained to produce interleaved multimodal reasoning traces:

| Token or action | Role |
| --- | --- |
| `<think>` | Textual reasoning and uncertainty assessment |
| `<tool_call>` or `Action` | Invoke image tools for segmentation or zoom-in |
| `<obs>` | Inject tool outputs, masks, crops, or overlays into the context |
| Final answer | Produce an answer grounded in accumulated visual evidence |

The model is not only optimized for answer accuracy. It is also encouraged to use tools when evidence is missing, avoid unnecessary calls when the answer is clear, and correct itself when tool observations contradict earlier reasoning.

---

## 🧰 Toolbox

Ophiuchus uses three complementary tools.

| Tool | Input | Output | Best used for |
| --- | --- | --- | --- |
| **SAM2** | Bounding boxes proposed by the MLLM | Segmentation masks | Fast region grounding when object location is roughly known |
| **BiomedParse** | Text prompts | Text-driven segmentation masks | Robust localization when bounding boxes are uncertain |
| **Zoom-In** | Bounding boxes or masks | Cropped regions and optional contour overlays | Fine-grained inspection of lesion morphology and boundaries |

In practice, Ophiuchus learns to mix these tools according to case difficulty and the type of missing evidence.

---

## 🩺 Case Showcase

Below are representative qualitative cases. More details, captions, and examples are provided in the paper.

### 1. Visual-Cue Search

<div align="center">
  <img src="asset/case_visual_cue_search.png" width="650" alt="Visual cue search case" />
</div>

**Pattern:** the model recognizes that the global view is insufficient, calls segmentation to locate candidate regions, then zooms in to verify subtle visual cues before answering.

### 2. Confirmation via Zoom-In

<div align="center">
  <img src="asset/case_confirmation.png" width="650" alt="Confirmation case" />
</div>

**Pattern:** after forming a tentative hypothesis, the model performs targeted zoom-ins to confirm key visual signatures and reduce uncertainty.

### 3. Hallucination Mitigation

<div align="center">
  <img src="asset/case_mitigation.png" width="650" alt="Hallucination mitigation case" />
</div>

**Pattern:** the model detects a mismatch between textual reasoning and visual evidence, invokes tools again, and revises the answer with better grounding.

---

## 🏁 Results

**Performance comparison on medical VQA benchmarks.** Gray-shaded rows denote large-sized models. **Bold** and <u>underlined</u> indicate the best and second-best results. *Improvement* denotes the absolute gain of **Ophiuchus** over Qwen2.5-VL-7B without tool use. Avg. is the arithmetic mean over seven out-of-domain zero-shot benchmarks. Since **Med-R1-2B** is trained on part of the OmniMedVQA test set, its Avg. is computed over the remaining six benchmarks.

<div align="center">
  <img width="880" alt="Ophiuchus benchmark results" src="asset/result.png">
</div>

---

## 📈 Citation and Growth

If Ophiuchus helps your research, please cite our paper and star the repository. Citation statistics will be updated after indexing by major academic search engines.

<a href="https://www.star-history.com/?repos=SII-zyj%2FOphiuchus&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=SII-zyj/Ophiuchus&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=SII-zyj/Ophiuchus&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=SII-zyj/Ophiuchus&type=date&legend=top-left" />
 </picture>
</a>

---

## 🧪 Data Preparation

This repository provides training scripts and a minimal data format example.

### SFT data: tool-integrated trajectories

A minimal example is provided here:

```text
./data/example/data_case/biomedparser-SFT.json
```

### RL data: one-case schema example

RL data is a list of records. A simplified case schema is shown below.

```json
{
  "question_id": 1,
  "question_type": "multiple choice | segmentation | open-ended | yes/no",
  "question": "string (the prompt/question)",
  "answer": "string (GT label or text answer)",
  "options": ["string list (only for MCQ / constrained answers)"],
  "data_type": "vqa | seg",
  "image_path": ["string list (one or multiple images)"],
  "source": "string (e.g., BioMedParse)",
  "mask_path": ["string list (only for seg tasks, optional)"],
  "max_steps": 6
}
```

---

## 🏋️ Training

### Installation for verl

```bash
# Please install verl from our repository.
# This helps visualize reward-component trends during training.
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation

pip3 install -e .
pip3 install vllm==0.8.5
```

### Install Toolbox Environments

> **Important:** To run an agent with external tools, install and configure the corresponding tool environment first. We recommend installing each tool in a separate conda environment to avoid package conflicts.

<details>
<summary><b>1. SAM2 environment</b></summary>

SAM2 requires `python>=3.10`, `torch>=2.5.1`, and `torchvision>=0.20.1`. Please follow the official PyTorch instructions to install PyTorch and TorchVision.

```bash
cd ./examples/env_server/sam2/sam2
pip install -e .
```

Download model checkpoints:

```bash
cd ./examples/env_server/sam2/checkpoints && \
./download_ckpts.sh && \
cd ..
```

Individual checkpoints:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

</details>

<details>
<summary><b>2. BiomedParse environment</b></summary>

Create a dedicated environment:

```bash
cd ./examples/env_server/BiomedParse/BiomedParse-main
conda create -n biomedparse python=3.9.19
conda activate biomedparse
```

Install PyTorch:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Install dependencies:

```bash
pip install -r assets/requirements/requirements.txt
```

Please download the Hugging Face model [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) into `/openai/clip-vit-base-patch32`, and download [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) into `BiomedParse-main/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`.

</details>

### Start training

First, run cold-start SFT and self-reflection fine-tuning. Please specify configurations in:

```text
/scripts/SFT/cold-start/my_qwen2_5vl_lora_sft.yaml
/scripts/SFT/reflective_rejection_fine_tuning/config_reflective.yaml
```

Update the `dataset` and `model_name_or_path` fields before training.

#### Stage A: Cold-start SFT

```bash
bash ./scripts/SFT/cold-start/train_cold_start.sh
```

#### Stage B: Self-reflection fine-tuning

```bash
bash ./scripts/SFT/reflective_rejection_fine_tuning/train_reflective.sh
```

#### Stage C: Agentic Tool RL, ATRL

Start the external toolbox first as local API services.

**Start the SAM2 server:**

```bash
bash ./examples/env_server/start_sam2_server.sh
```

**Start the BiomedParse server:**

```bash
bash ./examples/env_server/start_biomedparse_server.sh
```

Run GRPO training:

```bash
# Customize training parameters according to available hardware to avoid OOM.
bash ./examples/grpo_trainer/run_ATRL.sh
```

---

## 🤔 Model

| Model | Base model | Checkpoint | GPU memory |
| --- | --- | --- | --- |
| Ophiuchus | [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B) | Coming soon | 17GB |

---

## 🧑‍⚖️ Evaluation

Detailed evaluation instructions will be updated with the full release. The current release focuses on training scripts, tool environments, and the minimal data schema.

---

## 🐎 TODO

- [x] Announce ICML 2026 acceptance.
- [x] Add arXiv, PDF, GitHub, and Hugging Face paper links.
- [ ] Release model checkpoints on Hugging Face.
- [ ] Release the dataset on Hugging Face.
- [ ] Add complete evaluation scripts and reproduction commands.
- [ ] Add more qualitative examples and captions.
- [ ] Update citation statistics after indexing.

---

## 🙏 Acknowledgement

We gratefully acknowledge the inspiring work of [VERL](https://github.com/volcengine/verl), [verl-agent](https://github.com/volcengine/verl-agent), [SAM2](https://github.com/facebookresearch/sam2), and [BioMedParse](https://github.com/microsoft/BiomedParse). These projects provided essential foundations and inspiration for Ophiuchus. We also thank the open-source community for building the tools that make reproducible medical AI research possible.

---

## 📖 Citation

If you find Ophiuchus useful, please cite our paper.

```bibtex
@misc{jiang2025incentivizing,
  title         = {Incentivizing Tool-augmented Thinking with Images for Medical Image Analysis},
  author        = {Yankai Jiang and Yujie Zhang and Peng Zhang and Yichen Li and Jintai Chen and Xiaoming Shi and Shihui Zhen},
  year          = {2025},
  eprint        = {2512.14157},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2512.14157}
}
```
