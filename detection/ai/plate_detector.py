import cv2
import supervision as sv
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)

plate_model = YOLO("lpr.pt")  # License Plate YOLO model

PLATE_CLASS_ID = 0

def detect_plates(frame):
    plate_results = plate_model(frame, conf=0.4)[0]
    plate_dets = sv.Detections.from_ultralytics(plate_results)

    plates = []
    for xyxy, cls_id, conf in zip(plate_dets.xyxy, plate_dets.class_id, plate_dets.confidence):
        if int(cls_id) == PLATE_CLASS_ID:
            plates.append(list(map(int, xyxy)))

    return plates, plate_dets