import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from .preprocess import ImagePreprocessor
from .postprocess import ResultPostprocessor


class EmotionInference:
    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None,
                 conf_threshold: float = 0.5):
        """
        初始化表情识别推理器

        参数:
            model_path: 模型权重路径(.pt)
            device: 指定推理设备('cuda'/'cpu')
            conf_threshold: 识别置信度阈值
        """
        self.device = self._select_device(device)
        self.model = self._load_model(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = ['angry', 'happy', 'neutral', 'sad', 'surprise']
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = ResultPostprocessor()

    def _select_device(self, device: Optional[str]) -> str:
        """自动选择可用设备"""
        if device is None:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载YOLOv8模型"""
        try:
            model = torch.hub.load('ultralytics/yolov8',
                                   'custom',
                                   path=model_path,
                                   force_reload=True)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def predict(self, img: np.ndarray, conf_thresh: float = 0.5) -> List[Dict]:
        try:
            # 输入验证
            if img is None or not isinstance(img, np.ndarray):
                return []

            # 统一图像格式
            if len(img.shape) == 2:  # 灰度图转BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # 带透明度通道
                img = img[:, :, :3]

            # 模型推理
            results = self.model(img, imgsz=224, conf=conf_thresh)

            # 结果解析
            detections = []
            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue

                boxes = result.boxes.cpu().numpy()  # 确保转为numpy
                for box in (boxes if boxes.ndim > 1 else [boxes]):  # 处理单检测情况
                    detections.append({
                        'bbox': box.xywh[0].tolist() if box.xywh.ndim > 1 else box.xywh.tolist(),
                        'label': self.classes[int(box.cls)],
                        'conf': float(box.conf),
                    })
            return detections
        except Exception as e:
            print(f"[ERROR] Prediction failed: {str(e)}")
            return []

    def warmup(self, img_size=(640, 640)):
        """GPU预热"""
        if 'cuda' in self.device:
            img = torch.zeros((1, 3, *img_size),
                              device=self.device)
            for _ in range(3):
                self.model(img)
