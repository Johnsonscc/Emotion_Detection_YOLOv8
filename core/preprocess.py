# preprocess.py
import cv2
import numpy as np


def dynamic_resize(img, target_size=640):
    """保持宽高比的动态缩放"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def torch_normalize(img):
    """PyTorch风格的归一化处理"""
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std


def to_chw(img):
    """HWC转CHW格式适配PyTorch"""
    return np.transpose(img, (2, 0, 1)) if len(img.shape) == 3 else img


def preprocess_image(image, target_size=640):
    # 输入标准化处理
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # 新增预处理管线（总改动行数<15）
    image = dynamic_resize(image, target_size)  # 尺寸适配
    image = torch_normalize(image)  # 归一化
    return to_chw(image)  # 格式转换
