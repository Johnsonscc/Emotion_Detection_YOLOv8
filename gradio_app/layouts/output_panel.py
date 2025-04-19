import gradio as gr
import pandas as pd


def create() -> dict:
    """创建结果展示面板"""
    with gr.Column(variant="panel"):
        # 可视化结果
        result_image = gr.Image(
            label="检测结果",
            interactive=False
        )

        # 结构化数据
        with gr.Tab("数值结果"):
            result_table = gr.Dataframe(
                headers=["表情", "置信度", "位置"],
                datatype=["str", "number", "str"],
                row_count=5
            )

        # 历史统计
        with gr.Tab("趋势分析"):
            plot = gr.Plot(label="表情分布变化")

    return {
        "result_image": result_image,
        "result_table": result_table,
        "plot": plot
    }