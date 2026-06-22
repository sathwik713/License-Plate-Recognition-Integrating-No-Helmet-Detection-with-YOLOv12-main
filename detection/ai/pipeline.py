import cv2
import supervision as sv
from .helmet_detector import detect_helmets
from .plate_detector import detect_plates
from .ocr import perform_ocr, preprocess_plate
from ..models import Detection
import logging
import os
from django.conf import settings

logging.basicConfig(level=logging.INFO)

def belongs_to_no_helmet(plate_box, helmet_box):
    px1, py1, px2, py2 = plate_box
    hx1, hy1, hx2, hy2 = helmet_box

    plate_center_x = (px1 + px2) // 2

    return (
        hy2 < py1 and  # plate below helmet
        hx1 - 60 < plate_center_x < hx2 + 60
    )

def process_frame(frame, video_source):
    detections = []

    # Detect helmets
    no_helmet_boxes, helmet_dets, helmet_labels = detect_helmets(frame)

    # Detect plates
    plates, plate_dets = detect_plates(frame)

    # Annotate
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)

    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, helmet_dets)
    annotated = label_annotator.annotate(annotated, helmet_dets, helmet_labels)

    for plate_box in plates:
        matched = False
        for nh_box in no_helmet_boxes:
            if belongs_to_no_helmet(plate_box, nh_box):
                matched = True
                break

        if matched:
            x1, y1, x2, y2 = plate_box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = preprocess_plate(crop)
            plate_text = perform_ocr(crop_rgb)
            plate_text = plate_text.replace(" ", "").replace("-", "")

            if plate_text:
                # Save detection
                det = Detection.objects.create(
                    helmet_status='No Helmet',
                    license_plate_text=plate_text,
                    video_source=video_source
                )
                detections.append(det)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"NO HELMET: {plate_text}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

    return annotated, detections