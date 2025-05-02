import cv2
import numpy as np
import threading


class CameraProcessor:
    def __init__(self):
        self.capture = None
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()

    def start_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                return "无法访问摄像头"
            self.running = True
            self._start_frame_reader()
            return "摄像头已开启"
        return "摄像头已在运行中"

    def _start_frame_reader(self):
        def reader():
            while self.running:
                ret, frame = self.capture.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self.lock:
                        self.latest_frame = frame

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

    def get_camera_frame(self):
        if self.running and self.latest_frame is not None:
            with self.lock:
                # 添加帧有效性验证
                if self.latest_frame.shape[0] > 0 and self.latest_frame.shape[1] > 0:
                    return self.latest_frame.copy()
        return None

    def stop_camera(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        return "摄像头已停止"
