#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_sam2_npz_bbox.py

Test sam2_server.py:
- Send image_path + bbox (XYXY pixel coordinates) + optional point prompts to /segment
- Receive a .npz (masks / scores / low_res_masks)
- Select the mask with the highest score
- Overlay the mask on the original image for visualization and save the result
"""

import io
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ================= Configuration =================

# FastAPI server address
SERVER_URL = "http://127.0.0.1:6060"

# Local image path (must match the server-side image_path)
IMAGE_PATH = "/your/path/to/SFT+RL/data_case/Images/liver_case_original_image_1.png"

# Detected bbox_2d, XYXY pixel coordinates
BBOX_2D = [210, 201, 250, 242]   # [x1, y1, x2, y2]

# Optional: point prompts (pixel coordinates)
# Example: use a single foreground point: CLICKLIST = [[320, 240]]; LABELS = [1]
# If you do not want to use points for now, set both CLICKLIST and LABELS to None
# CLICKLIST = [[320, 240]]   # or None
# LABELS    = [1]            # or None
CLICKLIST = None   # or None
LABELS    = None           # or None

# Output files
NPZ_PATH = "/your/path/to/verl-agent/examples/env_server/sam2/sam2_server/test/sam2_outputs.npz"
OVERLAY_PATH = "/your/path/to/verl-agent/examples/env_server/sam2/sam2_server/test/overlay_mask.png"


# ================= API call function =================

def call_sam2_segment(
    server_url: str,
    image_path: str,
    bbox_xyxy,
    clicklist=None,
    labels=None,
    multimask_output: bool = True,
    return_logits: bool = False,
):
    """Call the /segment endpoint and return masks/scores/low_res_masks."""
    url = server_url.rstrip("/") + "/segment"

    # Note: the FastAPI side uses a Pydantic model; None will be correctly parsed as null
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
        print("[ERR] HTTP request failed, status =", resp.status_code)
        print("[ERR] body  =", resp.text)
        raise e

    # Server returns binary .npz content
    buf = io.BytesIO(resp.content)
    data = np.load(buf, allow_pickle=False)

    masks = data["masks"]          # (C, H, W)
    scores = data["scores"]        # (C,)
    low_res_masks = data["low_res_masks"]  # (C, h, w)

    print(f"[INFO] masks shape: {masks.shape}, dtype: {masks.dtype}")
    print(f"[INFO] scores: {scores}")
    print(f"[INFO] low_res_masks shape: {low_res_masks.shape}")

    # Also save a local copy of the npz for later analysis
    np.savez_compressed(NPZ_PATH, masks=masks, scores=scores, low_res_masks=low_res_masks)
    print(f"[INFO] .npz saved to {NPZ_PATH}")

    return masks, scores, low_res_masks


# ================= Visualization function =================

def overlay_mask_on_image(
    image_path: str,
    mask: np.ndarray,
    bbox_xyxy=None,
    clicklist=None,
    save_path: str = None,
):
    """Overlay a 0/1 mask on the original image, optionally drawing the bbox and point prompts."""

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # (H, W, 3)

    if mask.shape != img_np.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {img_np.shape[:2]}")

    # Ensure mask is binary 0/1
    mask_bin = (mask > 0.5).astype(np.uint8)

    overlay = img_np.copy()
    overlay[mask_bin == 1] = [255, 0, 0]  # red

    alpha = 0.5
    vis = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(vis)
    ax.axis("off")
    ax.set_title("SAM2 Mask Overlay")

    # Draw bbox
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

    # Draw point prompts
    if clicklist is not None and len(clicklist) > 0:
        xs = [p[0] for p in clicklist]
        ys = [p[1] for p in clicklist]
        ax.scatter(xs, ys, s=40, marker="o", edgecolors="black", facecolors="cyan")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Overlay result saved to {save_path}")

    plt.show()


# ================= Main pipeline =================

def main():
    # 1. Call the service and get all candidate masks
    masks, scores, _ = call_sam2_segment(
        SERVER_URL,
        IMAGE_PATH,
        BBOX_2D,
        clicklist=CLICKLIST,
        labels=LABELS,
        multimask_output=True,
        return_logits=False,
    )

    # 2. Select the mask with the highest score (official recommended usage)
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]

    print(f"[INFO] Selected best mask: index={best_idx}, score={scores[best_idx]:.4f}")

    # 3. Overlay on the original image for visualization (draw bbox and points)
    overlay_mask_on_image(
        IMAGE_PATH,
        best_mask,
        bbox_xyxy=BBOX_2D,
        clicklist=CLICKLIST,
        save_path=OVERLAY_PATH,
    )


if __name__ == "__main__":
    main()
