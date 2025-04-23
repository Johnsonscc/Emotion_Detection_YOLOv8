import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


class ResultVisualizer:
    def __init__(self,
                 bbox_color: tuple = (0, 255, 0),
                 text_color: tuple = (0, 0, 255),
                 font_scale: float = 0.8,
                 thickness: int = 2):
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.font_scale = font_scale
        self.thickness = thickness

        self.COLOR_MAP = {
            "angry": (0, 0, 255),
            "happy": (0, 255, 0),
            "neutral": (255, 255, 0),
            "sad": (255, 0, 0),
            "surprise": (128, 0, 255)
        }

    def draw_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        try:
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()  # 将 Tensor 转为 NumPy 数组

            img = img.copy()
            for det in detections:
                bbox = det.get('bbox', [])
                if len(bbox) != 4 or not all(isinstance(x, (int, float)) for x in bbox):
                    continue

                x, y, w, h = map(int, bbox)
                color = self.COLOR_MAP.get(det['label'], self.bbox_color)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, self.thickness)
                cv2.putText(img, det['label'], (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                            self.text_color, self.thickness)
            return img
        except Exception as e:
            print(f"[ERROR] Visualization failed: {str(e)}")
            return self.error_image("Render error")

    def draw_fps(self, image: np.ndarray, fps: float) -> np.ndarray:
        cv2.putText(image,
                    f"FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2)
        return image

    def plot_histogram(self, detections: List[Dict]) -> plt.Figure:
        labels = [d['label'] for d in detections]
        unique, counts = np.unique(labels, return_counts=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(unique, counts,
                      color=[self.COLOR_MAP.get(l, (0.5, 0.5, 0.5)) for l in unique])

        ax.set_title('表情分布统计')
        ax.set_ylabel('出现次数')
        plt.xticks(rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')

        return fig

    def error_image(self, text: str) -> np.ndarray:
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(img, text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2)
        return img
