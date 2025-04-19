import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from ultralytics import YOLO
import torch


class EmotionDetector:
    """YOLOv8表情识别模型封装类"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Args:
            model_path: 模型权重路径
            device: 指定计算设备 (cuda/cpu)
        """
        self.model = self._load_model(model_path)
        self.device = self._select_device(device)
        self.model.to(self.device)
        self.classes = ['anger', 'fear', 'happy', 'neutral', 'sad']

    def _load_model(self, model_path: str) -> YOLO:
        """加载模型并验证"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        return YOLO(model_path)

    def _select_device(self, device: Optional[str]) -> str:
        """自动选择最佳计算设备"""
        if device is None:
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return device

    def predict(self, img: np.ndarray, conf_thresh: float = 0.5) -> List[Dict]:
        """执行表情识别预测
        Args:
            img: 输入图像 (BGR格式)
            conf_thresh: 置信度阈值
        Returns:
            List[Dict]: 检测结果列表，每个元素包含:
                - bbox: [x, y, w, h]
                - label: 表情类别
                - conf: 置信度
                - landmarks: 面部关键点(可选)
        """
        try:
            # 转换图像格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 执行预测
            results = self.model(
                img_rgb,
                imgsz=224,
                conf=conf_thresh,
                device=self.device
            )

            # 解析结果
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'bbox': box.xywh[0].tolist(),  # [x_center, y_center, w, h]
                        'label': self.classes[int(box.cls)],
                        'conf': float(box.conf),
                        'landmarks': self._get_landmarks(result) if hasattr(result, 'keypoints') else None
                    })
            return detections
        except Exception as e:
            raise RuntimeError(f"预测失败: {str(e)}")

    def _get_landmarks(self, result) -> Optional[List[List[float]]]:
        """提取面部关键点"""
        if result.keypoints is None:
            return None
        return [point.tolist() for point in result.keypoints.xy[0]]