import cv2
import numpy as np
from pathlib import Path

def validate_image(image):
    """输入验证和基本转换"""
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise ValueError(f"无法读取图像文件: {image}")
    elif not isinstance(image, np.ndarray):
        raise TypeError("输入必须是文件路径或numpy数组")

    # 确保3通道(BGR或RGB)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("需3通道图像(H,W,C)")
    return image


def dynamic_resize_pad(image, target_size=640):

    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 尺寸调整（优化插值方式）
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # 颜色空间转换（强制BGR->RGB）
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 中心填充
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_rgb

    return padded, (scale, pad_left, pad_top)


def preprocess_image(image, target_size=640):
    """主预处理入口"""
    # 输入验证
    image = validate_image(image)

    # 执行核心处理
    processed, meta = dynamic_resize_pad(image, target_size)

    # 数值标准化和维度转换
    normalized = processed.astype(np.float32) / 255.0  # [0,1]范围
    channel_first = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW

    return channel_first, meta

