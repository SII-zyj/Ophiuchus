#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam2_server.py

FastAPI + SAM2ImagePredictor（与官方推理流程一致）：
- 使用 Hydra 载入 config yaml → 得到 cfg
- 调用你本地的 build_sam2(cfg=cfg, ckpt_path=..., device=...) 构建模型
- 用 SAM2ImagePredictor 做 set_image + predict
- 返回官方同款输出：masks, scores, low_res_masks（打包为 .npz）

HTTP:
  POST /segment
    {
      "image_path": "...",
      "bbox": [x1, y1, x2, y2],        # 可选，XYXY 像素坐标
      "clicklist": [[x, y], ...],      # 可选
      "labels": [1, 0, ...],           # 可选
      "multimask_output": true,
      "return_logits": false
    }

  返回：application/octet-stream （.npz）
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


# ======================= 命令行参数 =======================
parser = argparse.ArgumentParser(description="SAM2 图像分割服务（官方风格 + Hydra cfg）")
parser.add_argument("--gpu", type=int, default=0, help="使用的 GPU 序号（默认 0）")
parser.add_argument("--port", type=int, default=6060, help="服务端口号（默认 6060）")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/your/path/to/sam2.1_hiera_large.pt",
    help="SAM2 checkpoint 路径",
)
parser.add_argument(
    "--config",
    type=str,
    default="/your/path/to/verl-agent/examples/env_server/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    help="SAM2 config yaml 绝对路径（Hydra 配置）",
)
args = parser.parse_args()


# ======================= 请求体定义 =======================
class SegmentRequest(BaseModel):
    image_path: str = Field(..., description="待分割图片路径（本地路径）")

    bbox: Optional[List[float]] = Field(
        None,
        description="目标 bbox，XYXY 像素坐标，如 [x1, y1, x2, y2]；可为空",
    )

    clicklist: Optional[List[List[float]]] = Field(
        None,
        description="点坐标列表 [[x1,y1], [x2,y2], ...]；可为空",
    )
    labels: Optional[List[int]] = Field(
        None,
        description="点标签列表 [1,0,...]（1=前景，0=背景），长度与 clicklist 一致；可为空",
    )

    multimask_output: bool = Field(
        True,
        description="是否预测多张掩码（C 张）",
    )
    return_logits: bool = Field(
        False,
        description="是否返回 logits（官方 return_logits）；False 时返回 threshold 后的 mask",
    )


# ======================= SAM2 封装 =======================
class SAM2Service:
    def __init__(self, gpu_id: int, ckpt_path: str, cfg_yaml: str):
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到可用 CUDA 设备。")

        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)
        self.device_str = f"cuda:{gpu_id}"

        ckpt_path = os.path.abspath(ckpt_path)
        cfg_yaml = os.path.abspath(cfg_yaml)
        self.ckpt_path = ckpt_path
        self.cfg_yaml = cfg_yaml

        print(f"[INFO] 使用设备: {self.device_str}")
        print(f"[INFO] 加载 SAM2 模型: cfg={self.cfg_yaml}, ckpt={self.ckpt_path}")

        self.predictor = self._build_predictor()

    def _build_predictor(self) -> SAM2ImagePredictor:
        """
        与你之前能正常工作的写法一致：
        - 用 Hydra 载入 cfg
        - 调用本地 build_sam2(cfg=..., ckpt_path=..., device=...)
        - 用 SAM2ImagePredictor 封装（后处理全部交给官方）
        """
        config_dir = os.path.dirname(self.cfg_yaml)
        config_name = os.path.basename(self.cfg_yaml)
        config_name = config_name.replace(".yaml", "").replace(".yml", "")

        print(f"[Hydra] 初始化配置目录: {config_dir}")
        print(f"[Hydra] 配置文件名: {config_name}")

        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=config_name)
            print(f"[Hydra] ✅ 成功加载配置: {config_name}")

        # 关键：这里用 cfg=cfg，而不是 config_file / model_cfg
        sam2_model = build_sam2(
            cfg=cfg,
            ckpt_path=self.ckpt_path,
            device=self.device,
        )

        predictor = SAM2ImagePredictor(sam2_model)
        return predictor

    def _autocast_ctx(self):
        """与官方 demo 一致，在支持 bf16 的 CUDA 上开启 autocast。"""
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
        与官方 ImagePredictor 用法一致：
          predictor.set_image(np_image)
          masks, scores, low_res_masks = predictor.predict(...)
        """
        if point_coords is not None and point_labels is None:
            raise ValueError("提供了 clicklist 但未提供 labels")
        if point_labels is not None and point_coords is None:
            raise ValueError("提供了 labels 但未提供 clicklist")
        if point_coords is not None and len(point_coords) != len(point_labels):
            raise ValueError("clicklist 与 labels 长度不一致")

        np_image = np.array(image_pil.convert("RGB"))

        with torch.inference_mode(), self._autocast_ctx():
            self.predictor.set_image(np_image)

            masks, scores, low_res_masks = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
                return_logits=return_logits,
                normalize_coords=True,  # 像素坐标 → 归一化
            )

        return masks, scores, low_res_masks


# ======================= FastAPI & 模型实例 =======================
app = FastAPI(title="SAM2 图像分割服务（官方 + Hydra cfg）")

sam2_service = SAM2Service(
    gpu_id=args.gpu,
    ckpt_path=args.checkpoint,
    cfg_yaml=args.config,
)


# ======================= 接口实现 =======================
@app.post("/segment", response_class=Response)
async def segment(request: SegmentRequest):
    try:
        # 1. 检查图片路径
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail=f"图片路径不存在: {request.image_path}")
        if not os.path.isfile(request.image_path):
            raise HTTPException(status_code=400, detail=f"路径不是有效文件: {request.image_path}")

        # 2. 打开图片
        try:
            img = Image.open(request.image_path).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法打开图片文件: {e}")

        # 3. 解析 prompts
        points_np: Optional[np.ndarray] = None
        labels_np: Optional[np.ndarray] = None
        box_np: Optional[np.ndarray] = None

        if request.clicklist is not None:
            if request.labels is None:
                raise HTTPException(status_code=400, detail="提供了 clicklist 但未提供 labels")
            if len(request.clicklist) != len(request.labels):
                raise HTTPException(status_code=400, detail="clicklist 与 labels 长度不一致")
            points_np = np.asarray(request.clicklist, dtype=np.float32)
            labels_np = np.asarray(request.labels, dtype=np.int32)

        if request.bbox is not None:
            if len(request.bbox) != 4:
                raise HTTPException(status_code=400, detail="bbox 必须是长度为 4 的 [x1, y1, x2, y2]")
            box_np = np.asarray(request.bbox, dtype=np.float32)

        if points_np is None and box_np is None:
            raise HTTPException(status_code=400, detail="至少提供 bbox 或 clicklist 之一")

        # 4. 调 SAM2
        masks, scores, low_res_masks = sam2_service.segment(
            image_pil=img,
            point_coords=points_np,
            point_labels=labels_np,
            box=box_np,
            multimask_output=request.multimask_output,
            return_logits=request.return_logits,
        )

        # 5. 打包为 .npz 返回
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
        raise HTTPException(status_code=500, detail=f"处理失败: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": sam2_service.device_str,
        "checkpoint": sam2_service.ckpt_path,
        "config": sam2_service.cfg_yaml,
    }


if __name__ == "__main__":
    print(f"SAM2 服务启动中... 使用 GPU: {args.gpu}, 端口: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
