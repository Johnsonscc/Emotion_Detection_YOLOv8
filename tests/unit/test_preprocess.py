import cv2
import numpy as np
import pytest
from pathlib import Path
from core.preprocess import preprocess, auto_white_balance

@pytest.fixture
def test_image():
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return img

def test_preprocess_output_shape(test_image):
    """测试预处理输出形状"""
    processed = preprocess(test_image)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32

def test_auto_white_balance(test_image):
    """测试白平衡效果"""
    balanced = auto_white_balance(test_image)
    assert balanced.shape == test_image.shape
    # 验证像素值变化
    assert not np.array_equal(balanced, test_image)

def test_preprocess_normalization(test_image):
    """测试归一化范围"""
    processed = preprocess(test_image)
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0

@pytest.mark.parametrize("size", [(256, 256), (320, 240)])
def test_resize_effect(size, test_image):
    """测试不同尺寸的调整"""
    processed = preprocess(test_image, target_size=size)
    assert processed.shape[:2] == size