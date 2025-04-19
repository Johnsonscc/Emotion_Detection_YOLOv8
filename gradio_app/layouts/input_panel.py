import gradio as gr
from components.state_manager import StateManager

def create() -> dict:
    """创建输入控制面板（包含动态切换功能）"""
    with gr.Column(variant="panel", elem_id="input_panel"):
        # ================= 输入源选择 =================
        input_source = gr.Radio(
            label="📡 输入源选择",
            choices=["实时摄像头", "图片上传", "视频文件"],
            value="实时摄像头",
            elem_id="input_source_radio",
            interactive=True,
            scale=2
        )

        input_components = gr.Column(variant="compact")

        # ================= 输入组件容器 =================
        with input_components:
            # 摄像头输入组件（初始可见）
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                interactive=True,  # 必须启用交互
                type="numpy",  # 必须设置为numpy数组
                label="实时摄像头视图",
                elem_id="webcam_input",
                visible=True,
                width=600,
                mirror_webcam=True  # 添加镜像翻转
            )
            # 图片上传组件（初始隐藏）
            upload_image = gr.Image(
                sources=["upload"],
                type="filepath",
                label="上传图片文件",
                elem_id="image_upload",
                visible=False,
                width=600,
                interactive=True
            )

            # 视频上传组件（初始隐藏）
            upload_video = gr.Video(
                sources=["upload"],
                label="上传视频文件",
                elem_id="video_upload",
                visible=False,
                width=600,
                interactive=True
            )

        # ================= 输入源切换事件绑定 =================
        input_source.change(
            fn=lambda mode: [
                gr.update(visible=mode == "实时摄像头"),
                gr.update(visible=mode == "图片上传"),
                gr.update(visible=mode == "视频文件")
            ],
            inputs=input_source,
            outputs=[webcam, upload_image, upload_video]
        )

        # ================= 高级设置区 =================
        with gr.Accordion("⚙️ 高级设置", open=False):
            with gr.Row():
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="置信度阈值"
                )
                iou_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.45,
                    step=0.05,
                    label="重叠度阈值"
                )

            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=["yolov8n（标准版）", "yolov8s（精简版）", "yolov8m（增强版）"],
                    value="yolov8n（标准版）",
                    label="模型选择",
                    interactive=True
                )

        # ================= 操作按钮组 =================
        with gr.Row(variant="panel"):
            # input_panel.py 修改按钮定义
            start_btn = gr.Button(
                "🎬 开始识别",
                variant="primary",
                interactive=True,  # 初始允许点击
                visible=True
            )

            stop_btn = gr.Button(
                "⏹️ 停止识别",
                variant="secondary",
                interactive=False  # 初始禁用
            )
            reset_btn = gr.Button(
                "🔄 重置输入",
                variant="secondary"
            )

        # ================= 状态同步逻辑 =================
        # 同步输入类型到状态管理器
        input_source.change(
            fn=lambda mode: [
                # start_btn:
                gr.update(
                    interactive=(mode != "视频文件"),
                    variant="primary" if mode != "视频文件" else "secondary"
                ),
                # stop_btn:
                gr.update(interactive=False),
                # reset_btn:
                gr.update(interactive=True)
            ],
            inputs=input_source,
            outputs=[start_btn, stop_btn, reset_btn]
        )

        # 动态禁用操作按钮
        input_source.change(
            fn=lambda mode: [
                gr.update(interactive=mode != "视频文件"),
                gr.update(interactive=False),
                gr.update(interactive=True)
            ],
            inputs=input_source,
            outputs=[start_btn, stop_btn, reset_btn]
        )

    return {
        "input_source": input_source,
        "webcam": webcam,
        "upload_image": upload_image,
        "upload_video": upload_video,
        "conf_slider": conf_slider,
        "iou_slider": iou_slider,
        "model_selector": model_selector,
        "start_btn": start_btn,
        "stop_btn": stop_btn,
        "reset_btn": reset_btn
    }
