import cv2
import numpy as np

def process_detection_results(results):
    boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
    confs = results.boxes.conf.cpu().numpy() if results.boxes else []
    class_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes else []

    detections = []
    for box, conf, class_id in zip(boxes, confs, class_ids):
        detections.append({
            'box': box.tolist(),
            'confidence': float(conf),
            'class_id': int(class_id),
            'class_name': results.names[class_id]
        })

    return detections
