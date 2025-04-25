import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional


def get_text_size(draw: ImageDraw, text: str, font: Optional[ImageFont.FreeTypeFont] = None):
    """兼容不同Pillow版本的获取文本尺寸"""
    try:
        if font:
            # Pillow 10.0.0+ 使用getbbox方法
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            # 旧的textsize方法
            return draw.textsize(text)
    except AttributeError:
        # 更早版本的兼容
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_detections(image, detections, class_colors=None):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # 尝试加载字体(使用系统默认字体)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)

    # 默认颜色映射
    if class_colors is None:
        class_colors = {
            'anger': (255, 0, 0),  # 红色
            'fear': (128, 0, 128),  # 紫色
            'happy': (0, 255, 0),  # 绿色
            'neutral': (255, 255, 0),  # 黄色
            'sad': (0, 0, 255)  # 蓝色
        }

    for det in detections:
        box = det['box']
        class_name = det['class_name']
        confidence = det['confidence']
        color = class_colors.get(class_name, (255, 0, 0))

        # 绘制边界框
        draw.rectangle(box, outline=color, width=2)

        # 构建标签文本
        label = f"{class_name} {confidence:.2f}"

        # 获取文本尺寸
        text_width, text_height = get_text_size(draw, label, font)

        # 绘制标签背景
        draw.rectangle(
            [box[0], box[1] - text_height - 4,
             box[0] + text_width + 4, box[1]],
            fill=color
        )

        # 绘制标签文本
        draw.text(
            (box[0] + 2, box[1] - text_height - 2),
            label,
            fill=(255, 255, 255),
            font=font
        )

    return np.array(image)
