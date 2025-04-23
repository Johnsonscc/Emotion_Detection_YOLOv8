import cv2
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from preprocess import ImagePreprocessor


class EmotionInference:
    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None,
                 conf_threshold: float = 0.5,
                 input_size: Tuple[int, int] = (640, 640)):

        self.class_names = ['angry', 'happy', 'neutral', 'sad', 'surprise']
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.device = self._select_device(device)
        self.model = self._load_model(model_path)

    def _select_device(self, device: Optional[str]) -> str:
        return device if device else 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.to(self.device)
            print(f"✅ 模型加载成功 | 设备: {self.device}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def predict(self, img: np.ndarray) -> List[Dict]:
        if img is None or img.size == 0:
            return []

        # 确保图像是 NumPy 格式
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()

        h, w = img.shape[:2]
        results = self.model.predict(
            source=img,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            show=False,
            save=False
        )

        detections = []
        for result in results:
            if result.boxes is None or result.boxes.shape[0] == 0:
                continue

            boxes = result.boxes.xywhn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x_center, y_center, width, height = boxes[i]
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w - 1))
                y2 = max(y1 + 1, min(y2, h - 1))

                detections.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'label': self.class_names[cls_ids[i]],
                    'conf': float(confs[i])
                })

        return detections