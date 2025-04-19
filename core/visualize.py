import cv2
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class ResultVisualizer:
    """专业级结果可视化工具"""

    COLOR_MAP = {
        'anger': (255, 0, 0),
        'fear': (128, 0, 128),
        'happy': (255, 255, 0),
        'neutral': (0, 255, 0),
        'sad': (0, 0, 255)
    }

    def __init__(self, font_scale: float = 0.7, thickness: int = 2):
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """在图像上绘制检测结果
        Args:
            img: 原始图像 (BGR格式)
            detections: 检测结果列表
        Returns:
            带标注的图像 (BGR格式)
        """
        img = img.copy()
        for det in detections:
            img = self._draw_single_detection(img, det)
        return img

    def _draw_single_detection(self, img: np.ndarray, detection: Dict) -> np.ndarray:
        """绘制单个检测结果"""
        color = self.COLOR_MAP[detection['label']]

        # 解析边界框 [x_center, y_center, w, h]
        x, y, w, h = detection['bbox']
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        # 1. 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.thickness)

        # 2. 绘制标签
        label = f"{detection['label']}: {detection['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      self.font_scale, self.thickness)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                    (255, 255, 255), self.thickness)

        # 3. 绘制关键点 (如果存在)
        if 'landmarks' in detection and detection['landmarks']:
            for (lx, ly) in detection['landmarks']:
                cv2.circle(img, (int(lx), int(ly)), 3, (0, 255, 255), -1)

        return img

    def plot_histogram(self, detections: List[Dict]) -> plt.Figure:
        """生成表情统计直方图"""
        labels = [d['label'] for d in detections]
        unique, counts = np.unique(labels, return_counts=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(unique, counts,
                      color=[self.COLOR_MAP[l] for l in unique])

        ax.set_title('表情分布统计')
        ax.set_ylabel('出现次数')
        plt.xticks(rotation=45)

        # 在柱子上方显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')

        return fig