import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tempfile
from pathlib import Path
import gradio as gr
import numpy as np
import torch
from PIL import Image
from gradio_app.layouts.header import create_header
from gradio_app.layouts.input_panel import create_input_panel
from gradio_app.layouts.output_panel import create_output_panel
from gradio_app.layouts.footer import create_footer
from gradio_app.components.state_manager import StateManager
from gradio_app.components.camera_processor import CameraProcessor
from gradio_app.config.theme_config import load_theme
from gradio_app.components.performance_monitor import PerformanceMonitor
from core.inference import EmotionDetector
from core.postprocess import process_detection_results
from core.visualize import draw_detections

# 在创建Blocks前加载主题
theme = load_theme()

model = EmotionDetector("../models/yolov8s-emo.pt")
state_manager = StateManager()
# 初始化摄像头处理器
camera_processor = CameraProcessor()
#初始化性能监控组件
perf_monitor = PerformanceMonitor()

def analyze_image(image_path):
    try:
        if not image_path:
            raise ValueError("请先上传图片")

        results = model.predict(image_path)
        detections = process_detection_results(results)
        state_manager.update_detections(detections)

        image = Image.open(image_path)
        visualized = draw_detections(image, detections)
        stats = state_manager.generate_stats()
        pie_chart = state_manager.generate_pie_chart()

        return visualized, stats, pie_chart, detections

    except Exception as e:
        print(f"分析图片时出错: {e}")
        return None, gr.DataFrame(), None, {"error": str(e)}


def analyze_video(video_path):
    try:
        if not video_path:
            raise ValueError("请先上传视频")

        # 创建临时文件保存处理后的视频
        output_dir = Path(tempfile.mkdtemp())
        output_path = str(output_dir / "output.mp4")

        # 处理视频
        detections = model.process_video(video_path, output_path=output_path)
        state_manager.update_detections(detections)

        # 生成统计信息
        stats = state_manager.generate_stats()
        pie_chart = state_manager.generate_pie_chart()
        pie_chart.update_layout(autosize=True)  # 确保自动调整尺寸

        return output_path, stats, pie_chart, detections

    except Exception as e:
        print(f"分析视频时出错: {e}")
        return None, gr.DataFrame(), None, {"error": str(e)}


def analyze_camera(frame, state_manager):
    perf_monitor.start_frame()
    try:
        if frame is None:
            return None, None, None, {"status": "等待摄像头画面"}

        # 转换格式
        if isinstance(frame, np.ndarray):
            frame_np = frame
        else:  # 如果是PIL图像
            frame_np = np.array(frame)

        # 执行分析
        results = model.predict(frame_np)
        detections = process_detection_results(results)
        state_manager.update_detections(detections)

        # 可视化结果
        visualized = draw_detections(frame_np, detections)
        stats = state_manager.generate_stats()
        pie_chart = state_manager.generate_pie_chart()

        return visualized, stats, pie_chart, detections

    except Exception as e:
        print(f"摄像头分析出错: {e}")
        return None, None, None, {"error": str(e)}

    finally:
            perf_monitor.end_frame()

#性能更新回调
def update_perf():
    current_fps = perf_monitor.fps
    cpu = perf_monitor.cpu_usage

    return f"{current_fps:.2f} FPS", f"{cpu:.1f}%"

def run_app():
    with gr.Blocks(
            title="YOLOv8 表情识别系统",
            theme=theme,
            css="assets/styles.css",
    ) as demo:
        create_header()

        with gr.Row():
            # 左侧输入面板 (40%宽度)
            with gr.Column(scale=4):
                input_components = create_input_panel()

            # 右侧输出面板 (60%宽度)
            with gr.Column(scale=6):
                output_components = create_output_panel(state_manager)

        create_footer()
        # 图片分析回调
        input_components["image_button"].click(
            fn=analyze_image,
            inputs=input_components["image_input"],
            outputs=[
                output_components["result_image"],
                output_components["stats_display"],
                output_components["pie_plot"],
                output_components["raw_output"]
            ]
        )
        # 视频分析回调
        input_components["video_button"].click(
            fn=analyze_video,
            inputs=input_components["video_input"],
            outputs=[
                output_components["result_video"],
                output_components["stats_display"],
                output_components["pie_plot"],
                output_components["raw_output"]
            ]
        )
        # 开启摄像头
        input_components["start_button"].click(
            fn=lambda: camera_processor.start_camera(),
            outputs=input_components["camera_status"]  # 需要添加状态显示组件
        )
        # 停止摄像头
        input_components["stop_button"].click(
            fn=lambda: camera_processor.stop_camera(),
            outputs=None
        )
        # 实时摄像头帧获取
        demo.load(
            fn=lambda: camera_processor.get_camera_frame(),
            inputs=None,
            outputs=input_components["camera_output"],
            every=0.1,  # 原始100ms => 调整为动态时间
            queue=True  # 添加队列控制
        )

        # 性能数据轮询
        demo.load(
            fn=update_perf,
            inputs=None,
            outputs=[
                output_components["fps_display"],
                output_components["ram_usage"]
            ],
            every=0.1 # 1秒刷新频率
        )
        # 实时分析回调
        input_components["camera_output"].change(
            fn=lambda img: analyze_camera(img, state_manager),
            inputs=input_components["camera_output"],
            outputs=[
                output_components["result_image"],
                output_components["stats_display"],
                output_components["pie_plot"],
                output_components["raw_output"]
            ],
            show_progress="hidden",
            queue = True  # 添加队列机制
        )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860 ,
        max_threads=4,# 匹配CPU核心数
        share=True
    )


if __name__ == "__main__":
    run_app()
