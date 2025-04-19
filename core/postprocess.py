from typing import List, Dict
import numpy as np


def filter_detections(
        detections: List[Dict],
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45
) -> List[Dict]:
    """过滤和合并检测结果
    Args:
        detections: 原始检测结果
        conf_thresh: 置信度阈值
        iou_thresh: 重叠度阈值
    Returns:
        过滤后的检测结果
    """
    # 1. 按置信度过滤
    valid_dets = [d for d in detections if d['conf'] >= conf_thresh]

    # 2. NMS处理
    if len(valid_dets) > 1:
        valid_dets = non_max_suppression(valid_dets, iou_thresh)

    return valid_dets


def non_max_suppression(detections: List[Dict], iou_thresh: float) -> List[Dict]:
    """非极大值抑制实现"""
    if not detections:
        return []

    # 按置信度降序排序
    detections.sort(key=lambda x: x['conf'], reverse=True)

    keep = []
    while detections:
        keep.append(detections[0])
        detections = [
            det for det in detections[1:]
            if bbox_iou(keep[-1]['bbox'], det['bbox']) < iou_thresh
        ]
    return keep


def bbox_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个边界框的IoU"""
    # 转换为中心点坐标到角点坐标
    box1 = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
    box2 = [box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]

    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (box1_area + box2_area - inter_area)