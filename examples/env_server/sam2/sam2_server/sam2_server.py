#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam2_server.py

FastAPI + SAM2ImagePredictor (consistent with the official inference flow):
- Load the Hydra config yaml -> obtain cfg
- Build the model via your local build_sam2(cfg=cfg, ckpt_path=..., device=...)
- Use SAM2ImagePredictor to run set_image + predict
- Return the official-style outputs: masks, scores, low_res_masks (packed as .npz)

HTTP:
  POST /segment
    {
      "image_path": "...",
      "bbox": [x1, y1, x2, y2],        # optional, XYXY pixel coordinates
      "clicklist": [[x, y], ...],      # optional
      "labels": [1, 0, ...],           # optional
      "multimask_output": true,
      "return_logits": false
    }

  Response: application/octet-stream (.npz)
    np.load(...):
      masks         -> (C, H, W)
      scores        -> (C,)
      low_res_masks -> (C, h, w)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional

from PIL import Image
import io
import os
import argparse
import traceback

import numpy as np
import torch
import uvicorn

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra


# ======================= Command-line args =======================
parser = argparse.ArgumentParser(description="SAM2 image segmentation service (official style + Hydra cfg)")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (default: 0)")
parser.add_argument("--port", type=int, default=6060, help="Service port (default: 6060)")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/your/path/to/sam2.1_hiera_large.pt",
    help="Path to the SAM2 checkpoint",
)
parser.add_argument(
    "--config",
    type=str,
    default="/your/path/to/verl-agent/examples/env_server/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    help="Absolute path to the SAM2 config yaml (Hydra config)",
)
args = parser.parse_args()


# ======================= Request body =======================
class SegmentRequest(BaseModel):
    image_path: str = Field(..., description="Path to the input image (local path)")

    bbox: Optional[List[float]] = Field(
        None,
        description="Target bbox in XYXY pixel coordinates, e.g. [x1, y1, x2, y2]; can be null",
    )

    clicklist: Optional[List[List[float]]] = Field(
        None,
        description="List of point coordinates [[x1,y1], [x2,y2], ...]; can be null",
    )
    labels: Optional[List[int]] = Field(
        None,
        description="List of point labels [1,0,...] (1=foreground, 0=background), same length as clicklist; can be null",
    )

    multimask_output: bool = Field(
        True,
        description="Whether to predict multiple masks (C masks)",
    )
    return_logits: bool = Field(
        False,
        description="Whether to return logits (official return_logits); if False, returns thresholded masks",
    )


# ======================= SAM2 wrapper =======================
class SAM2Service:
    def __init__(self, gpu_id: int, ckpt_path: str, cfg_yaml: str):
        if not torch.cuda.is_available():
            raise RuntimeError("No available CUDA device detected.")

        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)
        self.device_str = f"cuda:{gpu_id}"

        ckpt_path = os.path.abspath(ckpt_path)
        cfg_yaml = os.path.abspath(cfg_yaml)
        self.ckpt_path = ckpt_path
        self.cfg_yaml = cfg_yaml

        print(f"[INFO] Using device: {self.device_str}")
        print(f"[INFO] Loading SAM2 model: cfg={self.cfg_yaml}, ckpt={self.ckpt_path}")

        self.predictor = self._build_predictor()

    def _build_predictor(self) -> SAM2ImagePredictor:
        """
        Same approach as your previously working implementation:
        - Load cfg via Hydra
        - Call local build_sam2(cfg=..., ckpt_path=..., device=...)
        - Wrap with SAM2ImagePredictor (let the official code handle post-processing)
        """
        config_dir = os.path.dirname(self.cfg_yaml)
        config_name = os.path.basename(self.cfg_yaml)
        config_name = config_name.replace(".yaml", "").replace(".yml", "")

        print(f"[Hydra] Initializing config directory: {config_dir}")
        print(f"[Hydra] Config file name: {config_name}")

        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=config_name)
            print(f"[Hydra] ✅ Successfully loaded config: {config_name}")

        # Key point: pass cfg=cfg, not config_file / model_cfg
        sam2_model = build_sam2(
            cfg=cfg,
            ckpt_path=self.ckpt_path,
            device=self.device,
        )

        predictor = SAM2ImagePredictor(sam2_model)
        return predictor

    def _autocast_ctx(self):
        """Match the official demo: enable autocast bf16 on CUDA devices that support it."""
        import contextlib

        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def segment(
        self,
        image_pil: Image.Image,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
        multimask_output: bool = True,
        return_logits: bool = False,
    ):
        """
        Same as the official ImagePredictor usage:
          predictor.set_image(np_image)
          masks, scores, low_res_masks = predictor.predict(...)
        """
        if point_coords is not None and point_labels is None:
            raise ValueError("clicklist was provided but labels were not provided")
        if point_labels is not None and point_coords is None:
            raise ValueError("labels were provided but clicklist was not provided")
        if point_coords is not None and len(point_coords) != len(point_labels):
            raise ValueError("clicklist and labels must have the same length")

        np_image = np.array(image_pil.convert("RGB"))

        with torch.inference_mode(), self._autocast_ctx():
            self.predictor.set_image(np_image)

            masks, scores, low_res_masks = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
                return_logits=return_logits,
                normalize_coords=True,  # pixel coords -> normalized
            )

        return masks, scores, low_res_masks


# ======================= FastAPI & model instance =======================
app = FastAPI(title="SAM2 image segmentation service (official + Hydra cfg)")

sam2_service = SAM2Service(
    gpu_id=args.gpu,
    ckpt_path=args.checkpoint,
    cfg_yaml=args.config,
)


# ======================= Endpoint implementation =======================
@app.post("/segment", response_class=Response)
async def segment(request: SegmentRequest):
    try:
        # 1. Validate image path
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail=f"Image path does not exist: {request.image_path}")
        if not os.path.isfile(request.image_path):
            raise HTTPException(status_code=400, detail=f"Path is not a valid file: {request.image_path}")

        # 2. Load image
        try:
            img = Image.open(request.image_path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to open image: {e}")

        # 3. Parse prompts
        points_np: Optional[np.ndarray] = None
        labels_np: Optional[np.ndarray] = None
        box_np: Optional[np.ndarray] = None

        if request.clicklist is not None:
            if request.labels is None:
                raise HTTPException(status_code=400, detail="clicklist was provided but labels were not provided")
            if len(request.clicklist) != len(request.labels):
                raise HTTPException(status_code=400, detail="clicklist and labels length mismatch")
            points_np = np.asarray(request.clicklist, dtype=np.float32)
            labels_np = np.asarray(request.labels, dtype=np.int32)

        if request.bbox is not None:
            if len(request.bbox) != 4:
                raise HTTPException(status_code=400, detail="bbox must be a length-4 list [x1, y1, x2, y2]")
            box_np = np.asarray(request.bbox, dtype=np.float32)

        if points_np is None and box_np is None:
            raise HTTPException(status_code=400, detail="At least one of bbox or clicklist must be provided")

        # 4. Run SAM2
        masks, scores, low_res_masks = sam2_service.segment(
            image_pil=img,
            point_coords=points_np,
            point_labels=labels_np,
            box=box_np,
            multimask_output=request.multimask_output,
            return_logits=request.return_logits,
        )

        # 5. Pack into .npz and return
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            masks=masks,
            scores=scores,
            low_res_masks=low_res_masks,
        )
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="sam2_outputs.npz"'},
        )

    except HTTPException:
        traceback.print_exc()
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": sam2_service.device_str,
        "checkpoint": sam2_service.ckpt_path,
        "config": sam2_service.cfg_yaml,
    }


if __name__ == "__main__":
    print(f"Starting SAM2 service... GPU: {args.gpu}, port: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
