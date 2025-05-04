import cv2
import numpy as np
import threading

class CameraProcessor:
    def __init__(self):
        self.capture = None
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        self.mirror_mode = True  # 初始状态（True为镜像模式）
        self.frame_buffer = []  # 新增缓冲队列
        self.buffer_size = 8 # 匹配MPS最优批大小

    def start_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if cv2.__version__ >= '4.5.0':
                self.capture.set(cv2.CAP_PROP_HW_ORIENTATION, 0)  # 关闭硬件自动旋转
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
        if not self.running or self.latest_frame is None:
            return None

        with self.lock:
            # 保持原始镜像处理不变
            frame = cv2.flip(self.latest_frame, 1) if self.mirror_mode else self.latest_frame.copy()

            # 自动填充缓冲队列
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)

            return frame

    def stop_camera(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        return "摄像头已停止"
