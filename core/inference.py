import torch
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import cv2
import os
import sys
from pathlib import Path
from core.preprocess import preprocess_image
from core.postprocess import process_detection_results
from core.visualize import draw_detections

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
        self.model = original_model
        self.model.model = self.quantized_model
        self.model.to(device)
        self.batch_size = 4
        self.class_names = ['anger', 'fear', 'happy', 'neutral', 'sad']

    def predict(self, image):
        # 新增预处理步骤
        preprocessed, padding_info = preprocess_image(image)
        tensor = torch.from_numpy(preprocessed).unsqueeze(0).to(self.device)

        with torch.no_grad():
            results = self.model(tensor)  # 保持原始结果对象

        print(f"原始输出box数量: {len(results[0].boxes)}")
        print(f"原始输出置信度: {results[0].boxes.conf if results[0].boxes else '无'}")
        return self._adjust_coordinates(results[0], padding_info)

    def _adjust_coordinates(self, result, padding_info):
        scale, left_pad, top_pad = padding_info
        # 强制转换为numpy数组以保证类型统一
        original_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        # Numpy向量化运算调整坐标（不操作原张量）
        adjusted = original_boxes.copy()
        adjusted[:, 0::2] -= left_pad
        adjusted[:, 1::2] -= top_pad
        adjusted /= scale

        setattr(result, 'adjusted_coords', adjusted.astype(np.float32))  # 显式类型声明
        return result

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
