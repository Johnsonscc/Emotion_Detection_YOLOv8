import cv2
import numpy as np
from gradio import Image
import threading
import time
from queue import Queue

class CameraProcessor:
    def __init__(self):
        self.capture = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=1)
        self.worker_thread = None

    def start_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise RuntimeError("无法访问摄像头")

            self.is_running = True
            self.worker_thread = threading.Thread(target=self._capture_frames)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            return "摄像头已开启"
        return "摄像头已在运行中"

    def _capture_frames(self):
        while self.is_running:
            ret, frame = self.capture.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 更新最新帧，丢弃旧帧
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            self.frame_queue.put(frame_rgb)

            time.sleep(0.05)  # 控制帧率

    def get_camera_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    def stop_camera(self):
        self.is_running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        return "摄像头已停止"
