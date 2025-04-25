from core.inference import EmotionDetector
import os


class ModelLoader:
    def __init__(self):
        self.model = None

    def load_model(self, model_path="../models/yolov8l-emo.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        self.model = EmotionDetector(model_path)
        return "模型加载成功"
