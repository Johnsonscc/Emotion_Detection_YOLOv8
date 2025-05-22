import time
import psutil
from collections import deque


class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.time_queue = deque(maxlen=window_size)
        self.process = psutil.Process()
        self.frame_count = 0
        self._frame_start = None

    def start_frame(self):
        self._frame_start = time.perf_counter()  # 强制重置时间戳

    def end_frame(self):
        if self._frame_start is not None:  # 添加有效性判断
            duration = time.perf_counter() - self._frame_start
            self.time_queue.append(duration)
            self.frame_count += 1
            self._frame_start = None  # 重置时间戳

    @property
    def fps(self):
        if not self.time_queue:
            return 0.0
        return 1.0 / (sum(self.time_queue) / len(self.time_queue))

    @property
    def cpu_usage(self):
        return self.process.cpu_percent(interval=None)
