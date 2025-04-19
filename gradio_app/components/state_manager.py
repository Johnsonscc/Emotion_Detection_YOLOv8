import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppState:
    current_model: str
    is_running: bool = True
    frame_count: int = 0
    input_type: str = "摄像头"  # add
    running_flag: bool = False  # 新增运行状态标志

class StateManager:
    """应用状态管理器"""
    def __init__(self):
        self.state = AppState(
            current_model="yolov8n",
            running_flag=False  # 显式初始化
        )
        self.lock = threading.Lock()

    # 新增状态操作方法
    def toggle_running(self):
        with self.lock:
            self.state.running_flag = not self.state.running_flag
            return self.state.running_flag

    def switch_model(self, model_name: str):
        """线程安全的模型切换"""
        with self.lock:
            self.state.current_model = model_name

    def get_state(self) -> AppState:
        """获取当前状态快照"""
        with self.lock:
            return self.state

    def toggle_running(self):
        """切换运行状态"""
        with self.lock:
            self.state.running_flag = not self.state.running_flag
            return self.state.running_flag

    def set_input_type(self, input_type: str):
        with self.lock:
            self.state.input_type = input_type

    def get_input_type(self) -> str:
        with self.lock:
            return self.state.input_type