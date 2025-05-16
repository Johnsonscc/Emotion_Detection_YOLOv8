import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # 新增全局导入

class StateManager:
    def __init__(self):
        self.current_detections = []
    #更新当前检测结果缓存
    def update_detections(self, detections):
        self.current_detections = detections
    #生成表情统计摘要
    def generate_stats(self):
        if not self.current_detections:
            return pd.DataFrame(columns=["表情", "数量", "平均置信度"])

        df = pd.DataFrame({
            '表情': [d['class_name'] for d in self.current_detections],
            '置信度': [d['confidence'] for d in self.current_detections]
        })

        stats = df.groupby('表情', observed=True).agg(
            数量=('表情', 'count'),
            平均置信度=('置信度', 'mean')
        ).reset_index()

        return stats.round({'平均置信度': 3})
    #生成表情分布饼状图
    def generate_pie_chart(self):
        stats = self.generate_stats()
        print("统计数据:", stats)

        if stats.empty or len(stats) == 0:
            # 使用全局限定的go对象
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=["无数据"],
                values=[1],
                marker_colors=['lightgrey']

            ))
            fig.update_layout(showlegend=False)
            return fig

        try:
            # 强制转换为整数
            counts = [int(x) for x in stats['数量'].tolist()]
            labels = stats['表情'].tolist()

            # 使用graph_objects直接创建可视化对象
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=counts,
                    hole=0.3,
                    textinfo='percent+label+value',
                    insidetextorientation='radial',
                    marker=dict(
                        colors=['#FF9999', '#99FF99', '#66B2FF', '#FFCC99', '#D8BFD8'],
                        line=dict(color='white', width=2)
                    )
                )
            ])
            # 统一布局配置
            fig.update_layout(
                margin=dict(t=40, b=20, l=20, r=20),
                showlegend=False,
                uniformtext_minsize=12,
                uniformtext_mode='hide'
            )

            return fig

        except Exception as e:
            print("生成饼图时出错:", e)
            import traceback
            traceback.print_exc()
            return None



