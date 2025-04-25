import gradio as gr


def create_input_panel():
    with gr.Tab("📁 输入方式", elem_classes=["input-tab"]):
        with gr.Tab("🖼️ 图片"):
            image_input = gr.Image(
                label="上传或拖放图片",
                type="filepath",
                height=300,
                interactive=True,
                elem_classes=["input-image"]
            )
            image_button = gr.Button(
                "🔍 分析图片",
                variant="primary",
                size="lg",
                elem_classes=["analyze-btn"]
            )

        with gr.Tab("🎬 视频"):
            video_input = gr.Video(
                label="上传或拖放视频",
                height=300,
                interactive=True,
                elem_classes=["input-video"]
            )
            video_button = gr.Button(
                "🔍 分析视频",
                variant="primary",
                size="lg",
                elem_classes=["analyze-btn"]
            )

        with gr.Tab("📷 实时摄像头"):
            gr.Markdown("### 🎥 实时表情识别")
            with gr.Row():
                camera_button = gr.Button(
                    "🎥 开启摄像头",
                    size="lg",
                    variant="secondary"
                )
                camera_stop = gr.Button(
                    "⏹️ 停止",
                    size="lg",
                    variant="stop"
                )
            camera_output = gr.Image(
                label="摄像头画面",
                streaming=True,
                height=300,
                elem_classes=["camera-output"]
            )

    gr.HTML("""
    <style>
        .input-tab {
            border-radius: 8px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
        }
        .input-image, .input-video, .camera-output {
            border-radius: 6px !important;
            border: 2px dashed var(--border-color-primary) !important;
        }
        .analyze-btn {
            margin-top: 12px !important;
            width: 100% !important;
        }
    </style>
    """)

    return {
        "image_input": image_input,
        "image_button": image_button,
        "video_input": video_input,
        "video_button": video_button,
        "camera_button": camera_button,
        "camera_stop": camera_stop,
        "camera_output": camera_output
    }
