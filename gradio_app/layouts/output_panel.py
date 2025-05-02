import gradio as gr


def create_output_panel(state_manager):
    with gr.Tabs(elem_classes=["output-tabs"]):
        with gr.Tab("🔍 检测结果"):
            with gr.Tabs():
                with gr.Tab("🖼️ 图片结果"):
                    result_image = gr.Image(
                        label="检测结果",
                        elem_classes=["result-image"],
                        height="auto",  # 关键修改点
                        container=True  # 去除默认容器约束
                    )
                with gr.Tab("🎬 视频结果"):
                    with gr.Column():
                        result_video = gr.Video(label="")
                        video_download = gr.Button(
                            "⬇️ 下载结果视频",
                            size="sm",
                            visible=False
                        )

        with gr.Tab("📊 统计信息", elem_id="stats-tab"):
            with gr.Row():
                with gr.Column(scale=7):
                    pie_plot = gr.Plot(
                        label="表情分布比例",
                        elem_id="pie-chart"
                    )
            stats_display = gr.DataFrame(
                headers=["表情", "数量", "平均置信度"],
                label="详细统计",
                interactive=False,
                elem_classes=["stats-table"]
            )

        with gr.Tab("📁 原始数据"):
            raw_output = gr.JSON(
                label="检测结果原始数据",
                elem_classes=["raw-data"]
            )

        # CSS样式
        gr.HTML("""
        <style>
            /* 图表容器样式 */
            #pie-chart, #bar-chart {
                border-radius: 8px;
                background: white;
                padding: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }

            /* 统计表格样式 */
            .stats-table {
                margin-top: 15px;
                max-height: 250px;
                overflow-y: auto;
            }

            /* 结果图片样式 */
            .result-image {
            max-width: 80% !important;
            height: auto !important;
            object-fit: contain !important;
            border-radius: 8px;
            }

            /* 标签样式 */
            .block-title {
                font-size: 1.1em !important;
                font-weight: 600 !important;
            }
            
        </style>
        """)
    return {
        "result_image": result_image,
        "result_video": result_video,
        "pie_plot": pie_plot,
        "stats_display": stats_display,
        "raw_output": raw_output,
        "video_download": video_download
    }

