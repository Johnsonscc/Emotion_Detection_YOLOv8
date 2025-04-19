from typing import List, Dict
import numpy as np


class ResultPostprocessor:
    def process(self,
                raw_output: np.ndarray,
                conf_threshold: float,
                class_names: List[str]) -> List[Dict]:
        """
        后处理流程：
        1. 按置信度过滤
        2. 非极大值抑制(NMS)
        3. 格式转换
        """
        # 初步过滤
        valid_detections = raw_output[raw_output[:, 4] > conf_threshold]

        if len(valid_detections) == 0:
            return []

        # 将坐标转换回原始图像坐标系
        converted = self._convert_coords(valid_detections)

        # NMS处理
        keep_indices = self._nms(converted[:, :4], converted[:, 4])

        outputs = []
        for idx in keep_indices:
            detection = valid_detections[idx]
            outputs.append({
                "bbox": detection[:4].tolist(),
                "conf": float(detection[4]),
                "class": class_names[int(detection[5])]
            })

        return outputs

    def _convert_coords(self, detections: np.ndarray) -> np.ndarray:
            """坐标解耦转换（根据预处理中的letterbox参数恢复）"""
            # 如果有使用letterbox预处理，需要对应逆向坐标变换
            # 此处示例仅做简单处理，实际需根据preprocess参数动态调整
            return detections


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