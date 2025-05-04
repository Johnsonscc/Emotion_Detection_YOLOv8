import torch
from ultralytics import YOLO
from tqdm import tqdm
import cv2
import os
import sys
from pathlib import Path


class EmotionDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'mps'):
        self.device = device
        # 修改模型加载方式
        original_model = YOLO(model_path)

        # 获取PyTorch原生模型
        torch_model = original_model.model

        # 动态量化配置
        self.quantized_model = torch.quantization.quantize_dynamic(
            torch_model.to('cpu'),  # 量化需要在CPU执行
            {torch.nn.Conv2d, torch.nn.Linear},  # 量化关键层
            dtype=torch.qint8
        ).to(device)

        # 保持与原YOLO类的兼容
        self.model = original_model
        self.model.model = self.quantized_model

        self.model.to(device)

        self.batch_size = 4
        self.class_names = ['anger', 'fear', 'happy', 'neutral', 'sad']

    def predict(self, image):
        results = self.model(image, verbose=False)
        return results[0]

    def process_video(self, video_path, output_path=None, fps=24):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_detections = []

        progress_bar = tqdm(total=total_frames, desc="Processing video")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.predict(frame_rgb)
                detections = process_detection_results(result)
                all_detections.extend(detections)

                if output_path:
                    visualized = draw_detections(frame_rgb, detections)
                    out.write(cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR))

                progress_bar.update(1)

        finally:
            cap.release()
            if output_path:
                out.release()
            progress_bar.close()

        return all_detections
