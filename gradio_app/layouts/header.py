import gradio as gr
from pathlib import Path


def create() -> dict:
    logo_path = Path(__file__).parent.parent / "assets/icons/logo.png"

    with gr.Row(variant="compact"):
        gr.Image(
            value=str(logo_path),  # 改用已定义的路径变量
            width=80,
            show_label=False,
            interactive=False,
            elem_id="header-logo"
        )

        # 中间标题
        with gr.Column():
            gr.Markdown("""
            # <center>基于YOLOv8的实时表情识别系统</center>
            > <center>版本 1.0 | 软件工程毕业设计</center>
            """)

        # 右侧控制按钮
        with gr.Column(scale=0):
            theme_btn = gr.Button("切换主题", variant="secondary")
            docs_btn = gr.Button("使用帮助", variant="primary")

    return {
        "theme_btn": theme_btn,
        "docs_btn": docs_btn
    }