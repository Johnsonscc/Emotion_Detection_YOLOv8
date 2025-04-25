import gradio as gr
from ..config.theme_config import load_theme

theme = load_theme()

def create_header():
    with gr.Row(variant="compact"):
        with gr.Column(scale=1):
            gr.Image("assets/icons/logo.png", width=80, show_label=False, show_download_button=False, container=False)
        with gr.Column(scale=5):
            gr.Markdown("""
            # <center>👨表情识别分析系统</center>
            ### <center>基于 YOLOv8 深度学习的情绪检测</center>
            """)
    gr.HTML("""<style>
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto;
        }
        .block-title {
            font-size: 1.2em !important;
        }
    </style>""")
