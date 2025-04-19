import pytest
from gradio_app.app import EmotionDetectionApp
from gradio import Blocks


class TestIntegration:
    @pytest.fixture
    def app(self):
        """论文需说明的测试框架搭建"""
        return EmotionDetectionApp()

    def test_build_interface(self, app):
        """界面构建完整性测试"""
        interface = app.build()
        assert isinstance(interface, Blocks)

        # 验证核心组件存在
        assert hasattr(interface, "input_panel")
        assert hasattr(interface, "output_panel")

    @pytest.mark.parametrize("input_type", ["image", "video"])
    def test_input_processing(self, app, input_type):
        """论文需包含的输入处理测试用例"""
        test_file = f"test_data/sample.{'jpg' if input_type == 'image' else 'mp4'}"
        result = app.process_input(test_file)
        assert "detections" in result