import torch
from ultralytics import YOLO
from tqdm import tqdm
import cv2
import os
import sys
from pathlib import Path

# 确保可以导入core模块中的其他文件
sys.path.append(str(Path(__file__).parent))
from postprocess import process_detection_results
from visualize import draw_detections

class EmotionDetector:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
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
