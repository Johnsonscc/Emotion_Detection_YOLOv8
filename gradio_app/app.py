import gradio as gr
import numpy as np
import cv2
from pathlib import Path
from typing import Iterator, Optional  # 新增导入
from layouts import assemble_layout
from components import (
    CameraProcessor,
    ResultRenderer,
    StateManager,
    ModelLoader
)
import yaml
class EmotionDetectionApp:
    """表情识别主应用类"""
    def __init__(self, config_path: str = "config/ui_config.yaml"):
        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化核心组件
        self.state = StateManager()
        self.model_loader = ModelLoader("config/model_config.yaml")
        self.camera = CameraProcessor(self.config["camera"])
        self.renderer = ResultRenderer()

        # 加载模型
        self.model = self.model_loader.load_model(
            Path("/Users/johnson/Desktop/Joh/Machine_Learning/Emotion_Detection_YOLOv8/models") / self.config["model"]["path"]
        )

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def create_interface(self):
        """构建Gradio界面"""
        with gr.Blocks(
                title=self.config["app"]["title"],
                css=self._get_css(),
                theme=gr.themes.Soft()
        ) as demo:
            # 1. 组装布局
            layout = assemble_layout(self.state)

            # 2. 事件绑定
            self._bind_events(layout)

        return demo

    def _get_css(self) -> str:
        """加载自定义CSS"""
        with open("assets/styles.css") as f:
            return f.read()

    def _bind_events(self, layout):
        """统一事件绑定方法"""
        # 摄像头实时处理
        layout["input_panel"]["webcam"].change(
            self._process_camera,
            inputs=layout["input_panel"]["webcam"],
            outputs=layout["output_panel"]["result_image"]
        )

        # 图片上传处理
        layout["input_panel"]["upload_image"].upload(
            self._process_image,
            inputs=layout["input_panel"]["upload_image"],
            outputs=layout["output_panel"]["result_image"]
        )

        # 视频上传处理
        layout["input_panel"]["upload_video"].upload(
            self._process_video,
            inputs=layout["input_panel"]["upload_video"],
            outputs=layout["output_panel"]["result_image"]
        )


    def _process_camera(self, frame: np.ndarray) -> np.ndarray:
        """处理摄像头连续输入"""
        if not self.state.state.running_flag:
            return None

        try:
            processed = self.camera.process_frame(frame)
            detections = self.model.predict(processed)
            return self.renderer.render(processed, detections)
        except Exception as e:
            print(f"Camera error: {str(e)}")
            return self.renderer.error_image()

    def _process_image(self, file_path: str) -> np.ndarray:
        """处理单张图片上传"""
        try:
            image = cv2.imread(file_path)
            detections = self.model.predict(image)
            return self.renderer.render(image, detections)
        except Exception as e:
            print(f"Image error: {str(e)}")
            return self.renderer.error_image()

    def _process_video(self, video_path: str) -> Iterator[Optional[np.ndarray]]:
        """处理视频逐帧解析（修复类型标注）"""
        cap = cv2.VideoCapture(video_path)

        # 添加视频文件打开校验
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频文件: {video_path}")
            yield self.renderer.error_image()
            return
        try:
            while self.state.state.running_flag:
                ret, frame = cap.read()
                if not ret:
                    break
                processed = self.camera.process_frame(frame)
                detections = self.model.predict(processed)
                yield self.renderer.render(processed, detections)
        finally:
            cap.release()

    def _process_frame(self, input_data):
        """处理视频帧的完整流水线（正确的方法位置）"""
        if not self.state.state.running_flag:  # 修正状态访问路径
            return None
        try:
            if isinstance(input_data, np.ndarray):  # 摄像头处理
                preprocessed = self.camera.process_frame(input_data)
            elif hasattr(input_data, "name"):  # 上传文件处理
                preprocessed = self._process_upload(input_data)
            else:
                raise ValueError("未知输入类型")
            detections = self.model.predict(preprocessed)
            return self.renderer.render(preprocessed, detections)
        except Exception as e:
            print(f"处理失败: {str(e)}")  # 替换错误日志方式
            return self.renderer.error_image()

    def _process_upload(self, file_obj):
        """处理文件上传（独立的方法定义）"""
        if file_obj.name.endswith(('.png', '.jpg', '.jpeg')):
            return cv2.imread(file_obj.name)
        elif file_obj.name.endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(file_obj.name)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        return None

    # 在 EmotionDetectionApp 类中添加
    def _check_button_state(self):
        """同步按钮状态的条件检查"""
        valid_model = self.model is not None
        valid_input = (
                self.state.state.input_type in
                ["实时摄像头", "图片上传", "视频文件"]
        )
        return gr.Button.update(
            interactive=valid_model and valid_input
        )


if __name__ == "__main__":
    app = EmotionDetectionApp()
    app.create_interface().launch(
        server_port=7860,
        share=False,
        show_error=True,    # 显示详细错误
        debug=True          # 调试模式
    )
