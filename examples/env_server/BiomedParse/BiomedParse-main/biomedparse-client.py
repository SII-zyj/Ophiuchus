import requests
import numpy as np
import io
from PIL import Image

# 服务端基础地址（默认端口8081）
BASE_URL = "http://localhost:8081"

def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    计算IoU（交并比）
    :param pred_mask: 预测掩码（float型，需二值化）
    :param gt_mask: 真值掩码（uint8型，0/1）
    :return: IoU值
    """
    # 预测掩码二值化（和官方示例一致，阈值0.5）
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    # 计算交集和并集
    intersection = (pred_binary & gt_mask).sum()
    union = (pred_binary | gt_mask).sum()
    # 避免除零
    if union == 0:
        return 0.0
    return intersection / union


def load_gt_mask(prompt: str) -> np.ndarray:
    """加载官方示例中的真值掩码"""
    gt_path = fr"examples/Part_1_516_pathology_breast_{prompt.replace(' ', '+')}.png"
    # 读取并转换为二值掩码
    gt_mask = Image.open(gt_path).convert('RGB')
    gt_mask = 1 * (np.array(gt_mask)[:, :, 0] > 0)
    return gt_mask

def main():
    # 官方示例
    IMAGE_PATH = r"examples/Part_1_516_pathology_breast.png"
    PROMPT = "neoplastic cells"

    # 发送分割请求
    try:
        request_data = {
            "image_path": IMAGE_PATH,
            "prompt": PROMPT
        }

        resp = requests.post(
            url=f"{BASE_URL}/segment",
            json=request_data,
            timeout=60
        )

        if resp.status_code == 200:
            # 解析预测掩码 掩码形状为(H,W)
            mask_buffer = io.BytesIO(resp.content)
            pred_mask = np.load(mask_buffer)
            print(f"分割结果：")

            # 4. 计算IoU和Dice
            # 加载真值掩码
            gt_mask = load_gt_mask(PROMPT)
            # 计算IoU
            iou = calculate_iou(pred_mask, gt_mask)
            pred_binary = (pred_mask > 0.5).astype(np.uint8)
            dice = (pred_binary & gt_mask).sum() * 2.0 / (pred_binary.sum() + gt_mask.sum())

            # 输出结果
            print(f"  IoU: {iou:.4f}")
            print(f"  Dice: {dice:.4f}")
        else:
            print(f"请求失败: {resp.status_code} - {resp.text}")

    except Exception as e:
        print(f"❌ 处理异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("===== BiomedParse 客户端示例=====")
    main()