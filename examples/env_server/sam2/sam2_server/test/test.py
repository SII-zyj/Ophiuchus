#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_sam2_npz_bbox.py

测试 sam2_server.py：
- 向 /segment 发送 image_path + bbox（XYXY 像素坐标）+ 可选点提示
- 收到 .npz（masks / scores / low_res_masks）
- 选择 scores 最大的那一张 mask
- 将 mask 叠加到原图上进行可视化，并保存结果
"""

import io
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ================= 配置区域 =================

# FastAPI 服务地址
SERVER_URL = "http://127.0.0.1:6060"

# 本地图片路径（和服务器那边 image_path 一致）
IMAGE_PATH = "/your/path/to/SFT+RL/data_case/Images/liver_case_original_image_1.png"

# 检测得到的 bbox_2d，XYXY 像素坐标
BBOX_2D = [210, 201, 250, 242]   # [x1, y1, x2, y2]

# 可选：点提示（像素坐标）
# 例如只用一个前景点：CLICKLIST = [[320, 240]]；LABELS = [1]
# 如果暂时不用点，把 CLICKLIST 和 LABELS 都设为 None 即可
# CLICKLIST = [[320, 240]]   # 或 None
# LABELS    = [1]            # 或 None
CLICKLIST = None   # 或 None
LABELS    = None           # 或 None

# 输出文件
NPZ_PATH = "/your/path/to/verl-agent/examples/env_server/sam2/sam2_server/test/sam2_outputs.npz"
OVERLAY_PATH = "/your/path/to/verl-agent/examples/env_server/sam2/sam2_server/test/overlay_mask.png"


# ================= 调接口函数 =================

def call_sam2_segment(
    server_url: str,
    image_path: str,
    bbox_xyxy,
    clicklist=None,
    labels=None,
    multimask_output: bool = True,
    return_logits: bool = False,
):
    """调用 /segment 接口，返回 masks/scores/low_res_masks"""
    url = server_url.rstrip("/") + "/segment"

    # 注意：FastAPI 那边是 pydantic 模型，None 会被正确解析成 null
    payload = {
        "image_path": image_path,
        "bbox": bbox_xyxy,
        "clicklist": clicklist,
        "labels": labels,
        "multimask_output": multimask_output,
        "return_logits": return_logits,
    }

    print(f"[INFO] POST {url}")
    print(f"[INFO] payload = {payload}")

    resp = requests.post(url, json=payload)
    try:
        resp.raise_for_status()
    except Exception as e:
        print("[ERR] HTTP 请求失败，status =", resp.status_code)
        print("[ERR] body  =", resp.text)
        raise e

    # 服务器返回的是 .npz 的二进制
    buf = io.BytesIO(resp.content)
    data = np.load(buf, allow_pickle=False)

    masks = data["masks"]          # (C, H, W)
    scores = data["scores"]        # (C,)
    low_res_masks = data["low_res_masks"]  # (C, h, w)

    print(f"[INFO] masks shape: {masks.shape}, dtype: {masks.dtype}")
    print(f"[INFO] scores: {scores}")
    print(f"[INFO] low_res_masks shape: {low_res_masks.shape}")

    # 也保存一份 npz 在本地，方便后续分析
    np.savez_compressed(NPZ_PATH, masks=masks, scores=scores, low_res_masks=low_res_masks)
    print(f"[INFO] .npz 已保存到 {NPZ_PATH}")

    return masks, scores, low_res_masks


# ================= 可视化函数 =================

def overlay_mask_on_image(
    image_path: str,
    mask: np.ndarray,
    bbox_xyxy=None,
    clicklist=None,
    save_path: str = None,
):
    """将 0/1 mask 叠加到原图上，可选画出 bbox 和点"""

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # (H, W, 3)

    if mask.shape != img_np.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} 与图像 shape {img_np.shape[:2]} 不匹配")

    # 确保 mask 是 0/1
    mask_bin = (mask > 0.5).astype(np.uint8)

    overlay = img_np.copy()
    overlay[mask_bin == 1] = [255, 0, 0]  # 红色

    alpha = 0.5
    vis = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(vis)
    ax.axis("off")
    ax.set_title("SAM2 Mask Overlay")

    # 画 bbox
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(rect)

    # 画点提示
    if clicklist is not None and len(clicklist) > 0:
        xs = [p[0] for p in clicklist]
        ys = [p[1] for p in clicklist]
        ax.scatter(xs, ys, s=40, marker="o", edgecolors="black", facecolors="cyan")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] 叠加结果已保存到 {save_path}")

    plt.show()


# ================= 主流程 =================

def main():
    # 1. 调用服务，拿到所有候选 masks
    masks, scores, _ = call_sam2_segment(
        SERVER_URL,
        IMAGE_PATH,
        BBOX_2D,
        clicklist=CLICKLIST,
        labels=LABELS,
        multimask_output=True,
        return_logits=False,
    )

    # 2. 选 scores 最大的那一张（官方推荐用法）
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]

    print(f"[INFO] 选取得分最高的 mask: index={best_idx}, score={scores[best_idx]:.4f}")

    # 3. 叠加到原图上进行可视化（画出 bbox 和点）
    overlay_mask_on_image(
        IMAGE_PATH,
        best_mask,
        bbox_xyxy=BBOX_2D,
        clicklist=CLICKLIST,
        save_path=OVERLAY_PATH,
    )


if __name__ == "__main__":
    main()
