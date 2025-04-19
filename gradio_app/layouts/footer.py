import gradio as gr
import datetime


def create() -> dict:
    """创建底部状态栏"""
    with gr.Row(variant="compact"):
        # 系统状态
        status = gr.HTML(
            value=f"""
            <div style='text-align: center'>
                <span>🟢 系统运行中</span> | 
                <span>CPU: 12%</span> | 
                <span>内存: 3.2/16GB</span> | 
                <span>更新时间: {datetime.datetime.now().strftime("%H:%M:%S")}</span>
            </div>
            """
        )

        # 版本信息
        version = gr.HTML(
            value="<div style='text-align: right'>版本 1.0.0 © 2023</div>"
        )

    return {
        "status": status,
        "version": version
    }