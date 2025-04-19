import cv2
import numpy as np
from typing import Tuple


def preprocess(
        img: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        do_clahe: bool = True
) -> np.ndarray:
    """标准化图像预处理流程
    Args:
        img: 输入图像 (BGR格式)
        target_size: 目标尺寸
        do_clahe: 是否使用自适应直方图均衡
    Returns:
        预处理后的图像 (RGB格式)
    """
    # 1. 自动白平衡
    img = auto_white_balance(img)

    # 2. 人脸对齐 (可选)
    img = align_face(img) if needs_alignment(img) else img

    # 3. 调整尺寸
    img = cv2.resize(img, target_size)

    # 4. 对比度增强
    if do_clahe:
        img = apply_clahe(img)

    # 5. 归一化
    img = img.astype(np.float32) / 255.0

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def auto_white_balance(img: np.ndarray) -> np.ndarray:
    """自动白平衡 (基于灰度世界假设)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    a = a - ((avg_a - 128) * 1.1)
    b = b - ((avg_b - 128) * 1.1)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """限制对比度自适应直方图均衡化"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def needs_alignment(img: np.ndarray) -> bool:
    """判断是否需要人脸对齐"""
    # 实现基于面部关键点检测的逻辑
    return False  # 简化示例