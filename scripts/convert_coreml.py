import coremltools as ct
from ultralytics import YOLO


def convert_to_coreml():
    # 加载原始模型
    model = YOLO('../models/yolov8s-emo.pt')

    # 使用样本数据校准转换
    sample_input = torch.rand(1, 3, 640, 640).cpu()  # YOLOv8输入尺寸

    # 转换配置
    model = ct.convert(
        model,
        inputs=[ct.ImageType(
            name="input",
            shape=sample_input.shape,
            scale=1 / 255.0,
            bias=[0, 0, 0]
        )],
        outputs=[ct.FieldName("output")],
        compute_units=ct.ComputeUnit.ALL,  # 自动选择ANE/GPU/CPU
        convert_to="mlprogram",
        # 优化神经引擎架构
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=False
    )

    # 添加元数据
    model.author = "Your Name"
    model.short_description = "YOLOv8 Facial Expression Detection"
    model.version = "1.0"

    # 保存模型
    model.save("../models/yolov8s-emo.mlpackage")
