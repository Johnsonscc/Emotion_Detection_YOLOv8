import matplotlib
matplotlib.use('Agg')  # 强制禁用 matplotlib 图形绘制
import gradio as gr
import numpy as np
import cv2
import yaml
import traceback
from pathlib import Path
from typing import Iterator, Optional
from layouts import assemble_layout
from components import (
    CameraProcessor,
    ResultRenderer,
    StateManager,
    ModelLoader
)

class EmotionDetectionApp:
    """表情识别主应用类（已修复参数不匹配错误）"""

    def __init__(self, config_path: str = "config/ui_config.yaml"):
        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化核心组件
        self.state = StateManager()
        self.model_loader = ModelLoader("config/model_config.yaml")
        self.camera = CameraProcessor(self.config["camera"])
        self.renderer = ResultRenderer()

        # 加载模型（确保使用正确的YOLO加载方式）
        model_path = Path("/Users/johnson/Desktop/Joh/Machine_Learning/Emotion_Detection_YOLOv8/models") / self.config["model"]["path"]
        self.model = self.model_loader.load_model(model_path)

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def create_interface(self):
        """构建Gradio界面（修复事件绑定）"""
        with gr.Blocks(
                title=self.config["app"]["title"],
                css=self._get_css(),
                theme=gr.themes.Soft()
        ) as demo:
            # 组装布局
            layout = assemble_layout(self.state)

            # 事件绑定（关键修复点）
            self._bind_events(layout)

        return demo

    def _get_css(self) -> str:
        """加载自定义CSS"""
        with open("assets/styles.css") as f:
            return f.read()

    def _bind_events(self, layout):
        """修复版事件绑定方法"""
        # ===== 实时摄像头流处理 =====
        layout["input_panel"]["webcam"].stream(  # 使用stream事件
            self._process_camera_frame,  # 处理单帧
            inputs=layout["input_panel"]["webcam"],
            outputs=layout["output_panel"]["result_image"]
        )

        # ===== 图片上传处理 =====
        layout["input_panel"]["upload_image"].upload(
            self._process_upload_image,  # 简化参数
            inputs=layout["input_panel"]["upload_image"],
            outputs=layout["output_panel"]["result_image"]
        )

        # ===== 视频上传处理 =====
        layout["input_panel"]["upload_video"].upload(
            self._process_video_stream,  # 使用正确的生成器
            inputs=layout["input_panel"]["upload_video"],
            outputs=layout["output_panel"]["result_image"]
        )

    def _process_camera_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理摄像头单帧输入（参数适配）"""
        if not self.state.state.running_flag:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            processed = self.camera.process_frame(frame)
            detections = self.model.predict(processed)
            return self.renderer.render(processed, detections)
        except Exception as e:
            print(f"Camera error: {str(e)}")
            return self.renderer.error_image()

    def _process_upload_image(self, file_path: str) -> np.ndarray:  # ✔️ 参数改为字符串
        """正确接收文件路径"""
        try:
            if not file_path:
                return self.renderer.error_image("空文件路径")

            # 直接使用字符串路径
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"无法读取图片: {file_path}")

            detections = self.model.predict(img)
            return self.renderer.render(img, detections)
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            return self.renderer.error_image()

    def _process_video_stream(self, video_path: str) -> Iterator[np.ndarray]:
        """生成器方式处理视频流（修复迭代问题）"""
        cap = cv2.VideoCapture(video_path)
        self.state.state.running_flag = True

        try:
            while self.state.state.running_flag and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed = self.camera.process_frame(frame)
                detections = self.model.predict(processed, verbose=True)  # 显式启用 verbose
                yield self.renderer.render(processed, detections)
        finally:
            cap.release()
            self.state.state.running_flag = False
            print("✅ 视频处理完成")

    def _check_button_state(self):
        """统一状态管理"""
        valid_model = self.model is not None
        valid_input = self.state.state.input_type in ["实时摄像头", "图片上传", "视频文件"]
        return gr.Button.update(interactive=valid_model and valid_input)


if __name__ == "__main__":
    app = EmotionDetectionApp()
    interface = app.create_interface()

    # 启动参数（推荐设置max_threads）
    try:
        # 你的主要代码逻辑
        interface.launch(
            server_port=7860,
            share=False,
            max_threads=4
        )
    except Exception as e:
        print(f"发生错误: {str(e)}")
        traceback.print_exc()