import torch
from ultralytics import YOLO
from pathlib import Path
from typing import Optional


class ModelLoader:
    """模型加载与管理组件"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = self._select_device()

    def load_model(self, model_path: Path) -> YOLO:
        """加载YOLOv8模型"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = YOLO(str(model_path))
        model.to(self.device)

        # 半精度加速
        if self.config["use_amp"]:
            model.half()

        return model

    def _load_config(self, path: str) -> dict:
        """加载模型配置"""
        # 实现配置文件读取逻辑
        return {
            "use_amp": True,
            "device": "auto"
        }

    def _select_device(self) -> str:
        """自动选择计算设备"""
        return "cuda" if torch.cuda.is_available() else "cpu"