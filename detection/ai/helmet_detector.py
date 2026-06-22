import cv2
import supervision as sv
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)

helmet_model = YOLO("best.pt")  # Helmet YOLO model

HELMET_LABEL_MAP = {
    0: "With Helmet",
    1: "Without Helmet"
}

def detect_helmets(frame):
    helmet_results = helmet_model(frame, conf=0.4)[0]
    helmet_dets = sv.Detections.from_ultralytics(helmet_results)

    no_helmet_boxes = []
    helmet_labels = []

    for xyxy, cls_id in zip(helmet_dets.xyxy, helmet_dets.class_id):
        label = HELMET_LABEL_MAP.get(int(cls_id), str(cls_id))
        helmet_labels.append(label)

        if label == "Without Helmet":
            no_helmet_boxes.append(list(map(int, xyxy)))

    return no_helmet_boxes, helmet_dets, helmet_labels