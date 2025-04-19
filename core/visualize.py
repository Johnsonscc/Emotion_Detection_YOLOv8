import cv2
import numpy as np
from typing import List, Dict


class ResultVisualizer:
    def __init__(self,
                 bbox_color: tuple = (0, 255, 0),
                 text_color: tuple = (0, 0, 255),
                 font_scale: float = 0.8,
                 thickness: int = 2):
        """
        可视化参数配置

        参数:
            bbox_color: 边界框颜色(BGR格式)
            text_color: 文字颜色
            font_scale: 字体大小
            thickness: 线条粗细
        """
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        try:
            # 输入验证
            if img is None or len(img.shape) != 3:
                return self.error_image("Invalid input")

            img = img.copy()
            if not detections:
                return img

            for det in detections:
                # 验证bbox格式
                bbox = det.get('bbox', [])
                if len(bbox) != 4 or not all(isinstance(x, (int, float)) for x in bbox):
                    continue

                # 转换为整数坐标
                x, y, w, h = map(int, bbox)

                # 验证坐标范围
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                if x + w > img.shape[1] or y + h > img.shape[0]:
                    continue

                # 绘制检测框
                cv2.rectangle(img, (x, y), (x + w, y + h), self.COLOR_MAP[det['label']], 2)

            return img
        except Exception as e:
            print(f"[ERROR] Visualization failed: {str(e)}")
            return self.error_image("Render error")

    def draw_fps(self, image: np.ndarray, fps: float) -> np.ndarray:
        """在图像左上角绘制FPS"""
        cv2.putText(image,
                    f"FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2)
        return image

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