import cv2
import numpy as np

def process_detection_results(results, conf_threshold=0.1):

    # 解包数据并进行设备转换
    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
    confs = results.boxes.conf.cpu().numpy() if results.boxes else []
    class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes else []

    detections = []
    for box, conf, class_id in zip(boxes, confs, class_ids):
        # 新增置信度过滤判断
        if conf >= conf_threshold:
            detections.append({
                'box': box.tolist(),  # 坐标转换为Python原生列表
                'confidence': float(conf),  # 显式转换为Python浮点数
                'class_id': int(class_id),  # 确保ID为整型
                'class_name': results.names[class_id]  # 同步获取类别名称
            })

    return detections