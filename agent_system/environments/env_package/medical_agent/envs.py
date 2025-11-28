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

from __future__ import annotations

import copy
import io
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gym
import numpy as np
import ray
import requests
from PIL import Image, ImageDraw

from agent_system.environments.prompts.medical_agent import (
    MEDICAL_AGENT_SYSTEM_PROMPT,
    MEDICAL_AGENT_TOOL_FEEDBACK,
    MEDICAL_AGENT_USER_PROMPT,
)

DEFAULT_TOOLS = ["Zoom-in", "SAM2", "BiomedParse"]
DEFAULT_TOOL_ENDPOINTS = {
    "BiomedParse": "/biomedparse",
    "SAM2": "/segment",
}


def _default_cases():
    return [
        {
            "question": "Based on this CT image, is there an abnormality?",
            "question_type": "multiple choice",
            "options": [
                "(A) pancreas tumor",
                "(B) No abnormality detected",
                "(C) liver tumor",
                "(D) kidney tumor",
                "(E) kidney cyst",
            ],
            "answer": "C",
            "data_type": "vqa",
            "source": "biomedparse",
            "image_path": [],
        },
    ]


def _read_image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.width, img.height


def _load_image_array(path: str) -> np.ndarray:
    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
        return np.array(rgb_img)


class MedicalToolClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def run(self, endpoint: str, payload: dict, expect_binary: bool = False):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        if expect_binary:
            return response.content

        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Tool server must return a JSON object.")
        return data


class MedicalAgentWorker:
    def __init__(self, seed: int, env_kwargs: dict | None = None):
        self._rng = np.random.RandomState(seed)
        self.env_kwargs = env_kwargs or {}
        self._cases = list(self.env_kwargs.get("cases", _default_cases()))
        if len(self._cases) == 0:
            self._cases = _default_cases()

        self._current_case: dict | None = None
        self._step_count = 0
        self._data_type = "vqa"
        self.max_steps = int(self.env_kwargs.get("max_steps", 4))
        self._done = False
        self._final_reward = 0.0
        self._won = False

        self.output_dir = Path(self.env_kwargs.get("output_dir", "./tmp/medical_agent"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mask_dir = self.output_dir / "masks"
        self.overlay_dir = self.output_dir / "overlays"
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)

        self.tool_timeout = int(self.env_kwargs.get("tool_timeout", 30))
        self.tool_endpoints = {**DEFAULT_TOOL_ENDPOINTS, **self.env_kwargs.get("tool_endpoints", {})}
        
        self._TOOL_BONUS = float(self.env_kwargs.get("tool_bonus", 2.0))  # B_bonus = 2 :contentReference[oaicite:3]{index=3}
        self._FORMAT_REWARD = 1.0

        # ===== Reward bookkeeping =====
        self._tool_used: bool = False             
        self._gt_mask_path: str | None = None     
        self._base_hw: tuple[int, int] | None = None  


        # Split tool base URLs
        default_tool_server_url = self.env_kwargs.get("tool_server_url", "http://0.0.0.0:6060").rstrip("/")
        default_tool_server_url_2 = self.env_kwargs.get("tool_server_url", "http://0.0.0.0:6061").rstrip("/")
        self.sam2_server_url = self.env_kwargs.get("sam2_server_url", default_tool_server_url).rstrip("/")
        self.biomedparse_server_url = self.env_kwargs.get("biomedparse_server_url", default_tool_server_url_2).rstrip("/")

        self._tool_clients = {
            "SAM2": MedicalToolClient(self.sam2_server_url, self.tool_timeout),
            "BiomedParse": MedicalToolClient(self.biomedparse_server_url, self.tool_timeout),
        }

        self._messages: list[dict[str, str]] = []
        self._images: list[dict[str, Any]] = []

    def _compute_format_reward(self) -> float:
        """
        S_format = 1 iff required fields appear and are in correct order.
        Keep Action-style tool calls (no <tool_call> required).
        We enforce:
        - each assistant message must contain <think>...</think>
        - each assistant message must be either:
            (a) a tool action via "Action: <ToolName>" (parsed by _parse_tool_invocations), OR
            (b) an answer (detected by _extract_answer)
        - the last assistant message must be an answer
        """
        assistant_msgs = [m["content"] for m in self._messages if m.get("role") == "assistant"]
        if not assistant_msgs:
            return 0.0

        think_re = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
        answer_block_re = re.compile(r"<answer>.*?</answer>", flags=re.IGNORECASE | re.DOTALL)

        # must end with an answer
        if not self._extract_answer(assistant_msgs[-1]):
            return 0.0

        for msg in assistant_msgs:
            if not think_re.search(msg):
                return 0.0

            is_answer = bool(self._extract_answer(msg))
            is_tool = bool(self._parse_tool_invocations(msg))  # relies on "Action:" + allowed tool names

            if not (is_answer or is_tool):
                return 0.0

            # Answer formatting: accept either <answer>...</answer> OR explicit "answer:" style
            if is_answer:
                has_structured_answer = bool(answer_block_re.search(msg))
                has_plain_answer = bool(re.search(r"answer\s*[:：]\s*[A-Za-z]", msg, flags=re.IGNORECASE))
                has_action_answer = bool(re.search(r"Action\s*:\s*Answer\b", msg, flags=re.IGNORECASE))
                if not (has_structured_answer or has_plain_answer or has_action_answer):
                    return 0.0

            # Tool formatting: must include "Action:" line (already implied by _parse_tool_invocations),
            # no <tool_call> tag required.
            if is_tool and not re.search(r"Action\s*:\s*[^\r\n]+", msg, flags=re.IGNORECASE):
                return 0.0

        return 1.0



    def _compute_iou_from_paths(self, pred_mask_path: str, gt_mask_path: str) -> float:
        if not self._base_hw:
            return 0.0
        H, W = self._base_hw
        pred = self._load_mask_array_from_path(pred_mask_path, (H, W)) > 0.5
        gt = self._load_mask_array_from_path(gt_mask_path, (H, W)) > 0.5
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return float(inter) / float(union) if union > 0 else 0.0


    def _compute_answer_reward(self, predicted_answer: str | None) -> float:
        """
        S_ans:
        - MCQ: exact match -> 1/0. :contentReference[oaicite:6]{index=6}
        - Seg: IoU piecewise -> 3/2/1/0. :contentReference[oaicite:7]{index=7}
        """
        # segmentation-like tasks: decide by data_type or availability of GT mask
        is_seg = ("seg" in (self._data_type or "").lower()) or (self._gt_mask_path is not None)
        if is_seg:
            if not self._gt_mask_path or not os.path.exists(self._gt_mask_path):
                return 0.0
            # choose the latest produced mask in trajectory
            pred_mask_path = None
            for meta in reversed(self._images):
                mp = meta.get("mask_path")
                if mp and os.path.exists(mp):
                    pred_mask_path = mp
                    break
            if not pred_mask_path:
                return 0.0

            iou = self._compute_iou_from_paths(pred_mask_path, self._gt_mask_path)
            if iou > 0.80:
                return 3.0
            if iou > 0.70:
                return 2.0
            if iou > 0.50:
                return 1.0
            return 0.0

        # multiple-choice / close-ended
        target = (self._current_case.get("answer") or "").upper() if self._current_case else ""
        return 1.0 if (predicted_answer and target and predicted_answer.upper() == target) else 0.0


    def _compute_tool_reward(self, s_ans: float) -> float:
        """
        S_tool: conditional bonus, only when S_ans > 0 and tool was used. Bonus fixed to 2. :contentReference[oaicite:8]{index=8}
        (Paper text also states: must invoke at least one external perception tool.) :contentReference[oaicite:9]{index=9}
        """
        if (s_ans > 0) and self._tool_used:
            return self._TOOL_BONUS
        return 0.0

    def _sample_case(self):
        idx = self._rng.randint(0, len(self._cases))
        return self._cases[idx]

    def _prepare_case(self, env_kwargs: dict | None) -> dict:
        if env_kwargs:
            return env_kwargs
        return self._sample_case()

    def _format_options(self, options: list | tuple | None) -> str:
        if not options:
            return ""
        return "\n".join(str(opt) for opt in options)

    def _resolve_image_path(self, case: dict) -> str:
        image_path = case.get("image_path") or []
        if isinstance(image_path, (list, tuple)):
            if len(image_path) == 0:
                raise ValueError("image_path is required for medical agent cases.")
            resolved = image_path[0]
        else:
            resolved = image_path
        resolved = os.path.abspath(resolved)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Image not found at {resolved}")
        return resolved

    def _copy_to_dir(self, src_path: str, dst_dir: Path, prefix: str) -> str:
        src_path = os.path.abspath(src_path)
        ext = Path(src_path).suffix or ".png"
        dst_path = dst_dir / f"{prefix}_{uuid.uuid4().hex}{ext}"
        shutil.copy2(src_path, dst_path)
        return str(dst_path.resolve())

    def _register_image(self, path: str, mask_path: str | None = None) -> dict:
        width, height = _read_image_size(path)
        meta = {
            "index": len(self._images) + 1,
            "path": path,
            "width": width,
            "height": height,
            "mask_path": mask_path,
        }
        self._images.append(meta)
        return meta

    def _build_user_prompt(self, case: dict, image_meta: dict) -> str:
        options_block = self._format_options(case.get("options"))
        return MEDICAL_AGENT_USER_PROMPT.format(
            question=case.get("question", ""),
            options=options_block,
            image_index=image_meta["index"],
            width=image_meta["width"],
            height=image_meta["height"],
        ).strip()

    def _format_tool_feedback(self, image_meta: dict) -> str:
        return MEDICAL_AGENT_TOOL_FEEDBACK.format(
            image_index=image_meta["index"],
            width=image_meta["width"],
            height=image_meta["height"],
        ).strip()

    def _compose_observation(self) -> Tuple[list[dict[str, str]], Any]:
        messages = [copy.deepcopy(msg) for msg in self._messages]
        if not self._images:
            return messages, None
        image_obs_list = [_load_image_array(meta["path"]) for meta in self._images]
        return messages, image_obs_list

    def reset(self, env_kwargs: dict | None = None):
        case = self._prepare_case(env_kwargs)
        self._current_case = case
        self._step_count = 0
        self._data_type = (case.get("data_type") or "vqa").lower()
        self.max_steps = int(case.get("max_steps", self.max_steps))
        self._done = False
        self._final_reward = 0.0
        self._won = False

        self._messages = []
        self._images = []
        
        self._tool_used = False

        gt_mask = case.get("mask_path")
        self._gt_mask_path = os.path.abspath(gt_mask) if gt_mask else None

        initial_image_path = self._resolve_image_path(case)
        image_meta = self._register_image(initial_image_path, case.get("mask_path"))
        self._base_hw = (image_meta["height"], image_meta["width"])

        self._messages.append({"role": "system", "content": MEDICAL_AGENT_SYSTEM_PROMPT.strip()})
        self._messages.append({"role": "user", "content": self._build_user_prompt(case, image_meta)})

        text_obs, image_obs = self._compose_observation()
        info = self._build_info(won=False)
        return text_obs, image_obs, info

    def _build_info(self, won: bool) -> dict:
        latest_mask = self._images[-1].get("mask_path") if self._images else None
        return {
            "available_tools": list(self._current_case.get("available_tools", DEFAULT_TOOLS)),
            "ground_truth": (self._current_case.get("answer") or "").upper(),
            "data_type": self._data_type,
            "question_id": self._current_case.get("question_id"),
            "question_type": self._current_case.get("question_type"),
            "source": self._current_case.get("source"),
            "image_path": [img["path"] for img in self._images],
            "mask_path": latest_mask,
            "won": won,
            "step_count": self._step_count,
        }

    def _extract_answer(self, action: str) -> str | None:
        for pattern in [r"<answer>\s*([A-Za-z])\b", r"answer\s*[:：]\s*([A-Za-z])"]:
            match = re.search(pattern, action, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().upper()
        letter_match = re.search(r"\b([A-E])\b", action, flags=re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
        return None

    def _get_image_by_index(self, index: int) -> dict:
        if index <= 0 or index > len(self._images):
            raise IndexError(f"Image index {index} is out of bounds.")
        return self._images[index - 1]

    @staticmethod
    def _strip_json_comments(s: str) -> str:
        s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
        s = re.sub(r"#.*?$", "", s, flags=re.MULTILINE)
        return s

    def _parse_tool_invocations(self, action: str) -> list[dict[str, Any]]:
        if not action:
            return []

        m = re.search(r"Action\s*:\s*([^\r\n]+)", action, flags=re.IGNORECASE)
        if not m:
            return []

        tool = m.group(1).strip()
        if tool.lower() == "answer":
            return []

        allowed = set(self._current_case.get("available_tools", DEFAULT_TOOLS))
        if tool not in allowed:
            return []

        tail = action[m.end() :]

        blocks = re.findall(r"```json\s*(.*?)\s*```", tail, flags=re.IGNORECASE | re.DOTALL)
        if not blocks:
            return [{"tool": tool, "payload": []}]

        for b in blocks:
            candidate = self._strip_json_comments(b).strip()
            if not candidate or candidate[0] not in "[{":
                continue
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                payload = [payload]
            if isinstance(payload, list):
                return [{"tool": tool, "payload": payload}]

        return [{"tool": tool, "payload": []}]

    def _call_segmenter(
        self,
        tool: str,
        image_path: str,
        extra_payload: dict,
        expect_binary: bool = False,
    ) -> str | bytes | None:
        endpoint = self.tool_endpoints.get(tool)
        if not endpoint:
            raise ValueError(f"No endpoint configured for tool {tool}")

        client = self._tool_clients.get(tool)
        if client is None:
            raise ValueError(f"No client configured for tool {tool}")

        payload = {"image": image_path, "image_path": image_path}
        payload.update(extra_payload)

        response = client.run(endpoint, payload, expect_binary=expect_binary)
        if isinstance(response, (bytes, bytearray)):
            return response
        return response.get("mask_path") or response.get("image_path")

    def _load_mask_array_from_path(self, mask_path: str, target_hw: Tuple[int, int]) -> np.ndarray:
        with Image.open(mask_path) as m:
            if m.mode == "RGBA":
                m = m.split()[-1].convert("L")
            else:
                m = m.convert("L")

            if m.size != (target_hw[1], target_hw[0]):
                m = m.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)

            arr = np.array(m).astype(np.float32) / 255.0
            return arr

    def _save_overlay_image(
        self,
        image_path: str,
        mask: np.ndarray,
        bbox_xyxy: List[float] | None = None,
        clicklist: List[List[float]] | None = None,
        save_path: str | None = None,
        alpha: float = 0.5,
    ) -> str:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img).astype(np.float32)
        h, w = img_np.shape[:2]

        if mask.shape != (h, w):
            mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
            mask = np.array(mask_img).astype(np.float32) / 255.0

        mask_bin = mask > 0.5
        overlay = img_np.copy()
        overlay[mask_bin] = np.array([255.0, 0.0, 0.0], dtype=np.float32)

        vis = (img_np * (1.0 - alpha) + overlay * alpha).clip(0, 255).astype(np.uint8)
        vis_img = Image.fromarray(vis)

        if save_path is None:
            save_path = str(self.overlay_dir / f"overlay_{uuid.uuid4().hex}.png")
        vis_img.save(save_path)
        return str(Path(save_path).resolve())

    def _handle_biomedparse(self, payload: dict) -> dict:
        idx = int(payload.get("index", 1))
        captions = payload.get("captions", "")
        base_image = self._get_image_by_index(idx)

        img_path = base_image["path"]
        tool_mask_path = self._call_segmenter("BiomedParse", img_path, {"captions": captions})
        if not tool_mask_path:
            raise RuntimeError("BiomedParse did not return a mask path.")
        tool_mask_path = os.path.abspath(tool_mask_path)

        try:
            mask_path = self._copy_to_dir(tool_mask_path, self.mask_dir, prefix="biomedparse_mask")
        except Exception as exc:  # noqa: BLE001
            print(f"!!(Warning) copy mask failed (BiomedParse): {type(exc).__name__}: {exc}")
            mask_path = tool_mask_path

        try:
            with Image.open(img_path) as im:
                h, w = im.height, im.width
            mask_arr = self._load_mask_array_from_path(mask_path, (h, w))
            overlay_path = str(self.overlay_dir / f"biomedparse_overlay_{uuid.uuid4().hex}.png")
            overlay_path = self._save_overlay_image(img_path, mask_arr, save_path=overlay_path, alpha=0.5)
            image_meta = self._register_image(overlay_path, mask_path=mask_path)
        except Exception as exc:  # noqa: BLE001
            print(f"!!(Warning) overlay failed for BiomedParse: {type(exc).__name__}: {exc}")
            image_meta = self._register_image(mask_path, mask_path=mask_path)

        return image_meta

    def _handle_sam2(self, payload: dict) -> dict:
        idx = int(payload.get("index", 1))
        bbox = payload.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            raise ValueError("SAM2 requires bbox_2d with four coordinates.")

        base_image = self._get_image_by_index(idx)
        img_path = base_image["path"]

        clicklist = payload.get("clicklist", None)
        labels = payload.get("labels", None)

        extra = {"bbox": bbox, "bbox_2d": bbox}
        if clicklist is not None:
            extra["clicklist"] = clicklist
        if labels is not None:
            extra["labels"] = labels
        if "multimask_output" in payload:
            extra["multimask_output"] = payload["multimask_output"]
        if "return_logits" in payload:
            extra["return_logits"] = payload["return_logits"]

        npz_bytes = self._call_segmenter("SAM2", img_path, extra, expect_binary=True)
        if not isinstance(npz_bytes, (bytes, bytearray)):
            raise RuntimeError("SAM2 did not return binary .npz content.")

        with np.load(io.BytesIO(npz_bytes)) as data:
            masks = data.get("masks")
            scores = data.get("scores")

        if masks is None:
            raise ValueError("SAM2 response is missing 'masks'.")

        if masks.ndim == 2:
            best_mask = masks.astype(np.float32)
        elif masks.ndim == 3:
            if scores is not None and len(scores) == masks.shape[0]:
                best_idx = int(np.argmax(scores))
            else:
                best_idx = 0
            best_mask = masks[best_idx].astype(np.float32)
        else:
            raise ValueError(f"Unexpected SAM2 mask shape: {masks.shape}")

        mask_bin = (best_mask > 0.5).astype(np.uint8) * 255
        mask_path = str(self.mask_dir / f"sam2_mask_{uuid.uuid4().hex}.png")
        Image.fromarray(mask_bin).save(mask_path)
        mask_path = str(Path(mask_path).resolve())

        overlay_path = str(self.overlay_dir / f"sam2_overlay_{uuid.uuid4().hex}.png")
        overlay_path = self._save_overlay_image(
            img_path,
            best_mask,
            bbox_xyxy=bbox,
            clicklist=clicklist,
            save_path=overlay_path,
            alpha=0.5,
        )

        image_meta = self._register_image(overlay_path, mask_path=mask_path)
        return image_meta

    def _clamp_bbox(self, bbox: list[float], width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), width))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height))
        y2 = max(0, min(int(y2), height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box after clamping.")
        return x1, y1, x2, y2

    def _handle_zoom_in(self, payload: dict) -> dict:
        idx = int(payload.get("index", 1))
        bbox = payload.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            raise ValueError("Zoom-in requires bbox_2d with four coordinates.")

        base_image = self._get_image_by_index(idx)
        img_path = base_image["path"]

        with Image.open(img_path) as img:
            w, h = img.width, img.height
            x1, y1, x2, y2 = self._clamp_bbox(bbox, w, h)
            cropped = img.crop((x1, y1, x2, y2))
            new_path = str(self.overlay_dir / f"zoom_{uuid.uuid4().hex}.png")
            cropped.save(new_path)

        image_meta = self._register_image(str(Path(new_path).resolve()), base_image.get("mask_path"))
        return image_meta

    def _execute_tool_action(self, tool: str, payload: dict) -> dict:
        if tool == "BiomedParse":
            return self._handle_biomedparse(payload)
        if tool == "SAM2":
            return self._handle_sam2(payload)
        if tool == "Zoom-in":
            return self._handle_zoom_in(payload)
        raise ValueError(f"Unsupported tool: {tool}")

    def step(self, action: str):
        if getattr(self, "_done", False):
            info = self._build_info(getattr(self, "_won", False))
            info.update({
                "tool_calling": False,
                "already_done": True,
                "final_reward": float(getattr(self, "_final_reward", 0.0)),
            })
            text_obs, image_obs = self._compose_observation()
            return text_obs, image_obs, 0.0, True, info

        self._step_count += 1
        action = action or ""
        self._messages.append({"role": "assistant", "content": action})

        reward = 0.0
        done = False
        won = False

        predicted_answer: str | None = None   
        step_info: Dict[str, Any] = {
            "tool_calling": False,
            "tool_group": "MedicalTools",
            "tool_name": None,
            "tool_input": None,
            "tool_error": None,
        }
        new_feedback: list[str] = []

        tool_invocations = self._parse_tool_invocations(action)

        if tool_invocations:
            step_info["tool_calling"] = True
            for item in tool_invocations:
                tool_name = item["tool"]
                step_info["tool_name"] = tool_name
                for payload in item.get("payload", []):
                    step_info["tool_input"] = payload
                    try:
                        image_meta = self._execute_tool_action(tool_name, payload)
                        self._tool_used = True  
                        new_feedback.append(self._format_tool_feedback(image_meta))
                    except Exception as exc:
                        err = (
                            f"!!(Warning) tool call failed: tool={tool_name}, input={payload}, "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        print(err)
                        step_info["tool_error"] = str(exc)
                        new_feedback.append(err)
        else:
            predicted_answer = self._extract_answer(action) 
            done = bool(predicted_answer)

        if new_feedback:
            for feedback in new_feedback:
                self._messages.append({"role": "user", "content": feedback})

        if self._step_count >= self.max_steps:
            done = True

        if done:
            s_format = self._compute_format_reward()
            s_ans = self._compute_answer_reward(predicted_answer)  
            s_tool = self._compute_tool_reward(s_ans)

            reward = float(s_ans + s_format + s_tool)
            won = bool(s_ans > 0)

            self._done = True
            self._won = won
            self._final_reward = reward

        text_obs, image_obs = self._compose_observation()
        info = self._build_info(won)
        info.update(step_info)
        return text_obs, image_obs, reward, done, info



class MedicalAgentEnvs(gym.Env):
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train

        env_worker = ray.remote(**resources_per_worker)(MedicalAgentWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), env_kwargs)
            self._workers.append(worker)

    def step(self, actions: list[str]):
        if len(actions) > self.num_processes:
            raise ValueError(f"Expected at most {self.num_processes} actions, got {len(actions)}")

        pad_n = self.num_processes - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        futures = []
        for worker, action in zip(self._workers, padded_actions):
            futures.append(worker.step.remote(action))
        results = ray.get(futures)

        text_list, image_list, reward_list, done_list, info_list = [], [], [], [], []
        for keep, result in zip(valid_mask, results):
            if not keep:
                continue
            obs, image_obs, reward, done, info = result
            text_list.append(obs)
            image_list.append(image_obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return text_list, image_list, reward_list, done_list, info_list

    def reset(self, kwargs: list[dict] | None = None):
        if kwargs is None:
            kwargs_list: list[dict] = []
        elif isinstance(kwargs, np.ndarray):
            kwargs_list = kwargs.tolist()
        else:
            kwargs_list = list(kwargs)

        if len(kwargs_list) > self.num_processes:
            raise ValueError(f"Expected at most {self.num_processes} env kwargs, got {len(kwargs_list)}")

        pad_n = self.num_processes - len(kwargs_list)
        padded_kwargs = list(kwargs_list) + [{}] * pad_n
        valid_mask = [True] * len(kwargs_list) + [False] * pad_n

        futures = []
        for worker, env_kwargs in zip(self._workers, padded_kwargs):
            futures.append(worker.reset.remote(env_kwargs))
        results = ray.get(futures)

        text_list, image_list, info_list = [], [], []
        for keep, result in zip(valid_mask, results):
            if not keep:
                continue
            obs, image_obs, info = result
            text_list.append(obs)
            image_list.append(image_obs)
            info_list.append(info)
        return text_list, image_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        for worker in self._workers:
            ray.kill(worker)
        self._closed = True

    def __del__(self):
        self.close()


def build_medical_agent_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict | None = None,
):
    return MedicalAgentEnvs(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
