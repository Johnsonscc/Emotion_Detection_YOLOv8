from core.visualize import draw_detections


class ResultRenderer:
    def __init__(self, class_colors=None):
        self.class_colors = class_colors or {
            'anger': (255, 0, 0),
            'fear': (128, 0, 128),
            'happy': (0, 255, 0),
            'neutral': (255, 255, 0),
            'sad': (0, 0, 255)
        }

    def render_results(self, image, detections):
        try:
            visualized = draw_detections(image, detections, self.class_colors)
            return visualized
        except Exception as e:
            print(f"渲染结果时出错: {e}")
            return image
