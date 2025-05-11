import cv2
import numpy as np
import os
import datetime

def dynamic_resize_pad(image, target_size=640, pad_color=(114, 114, 114)):
    """动态调整尺寸并保持宽高比，添加边缘填充"""
    h, w = image.shape[:2]
    print(f"原始尺寸: {w}x{h}")  # 调试用
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h))

    # 创建目标画布
    padded = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)

    # 计算填充位置
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    print(f"缩放后尺寸: {new_w}x{new_h} | 填充后尺寸: {target_size}x{target_size}")
    return padded, (scale, (left, top))


# 调整归一化逻辑（原代码有错误）
def torch_normalize(image):
    """PyTorch标准化处理 (修复像素值缩放错误)"""
    image = image.astype(np.float32) / 255.0#缩放
    mean = np.array([0.406, 0.456, 0.485], dtype=np.float32) * 255  # 转换为像素值范围
    std = np.array([0.225, 0.224, 0.229], dtype=np.float32) * 255  # 转换为像素值范围
    return (image - mean) / std  # 基于OpenCV的BGR格式处理


def preprocess_image(image, target_size=640):
    # 统一输入格式
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        if image.shape[-1] == 4:  # 处理RGBA格式
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # 执行预处理流程
    processed, (scale, (left_pad, top_pad)) = dynamic_resize_pad(image, target_size)
    normalized = torch_normalize(processed)
    return np.transpose(normalized, (2, 0, 1)), (scale, left_pad, top_pad)  # CHW格式