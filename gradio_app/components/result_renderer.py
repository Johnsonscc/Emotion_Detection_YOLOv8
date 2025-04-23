import cv2
import numpy as np
from typing import List, Dict


class ResultRenderer:
    """检测结果可视化渲染器"""

    COLOR_MAP = {
        'anger': (0, 0, 255),
        'fear': (128, 0, 128),
        'happy': (0, 255, 255),
        'neutral': (0, 255, 0),
        'sad': (255, 0, 0)
    }

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2

    def render(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """渲染检测结果到图像"""
        if image is None:
            return self.error_image()

        img = image.copy()
        for det in detections:
            img = self._draw_detection(img, det)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _draw_detection(self, img: np.ndarray, detection: Dict) -> np.ndarray:
        label = detection["label"]
        color = self.COLOR_MAP.get(label, (0, 255, 0))

        # 安全转换 bbox
        bbox = detection["bbox"]
        if hasattr(bbox, "cpu"):  # torch.Tensor
            bbox = bbox.cpu().numpy().tolist()
        x, y, w, h = map(int, bbox)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, self.thickness)

        text = f"{label}: {detection.get('conf', 0):.2f}"
        (tw, th), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        cv2.rectangle(img, (x, y - th - 5), (x + tw, y), color, -1)
        cv2.putText(img, text, (x, y - 5), self.font, self.font_scale, (255, 255, 255), self.thickness)

        return img

    def error_image(self) -> np.ndarray:
        """生成错误提示图像"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "ERROR", (220, 240),
                    self.font, 2, (0, 0, 255), 3)
        return img