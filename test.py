import cv2
import supervision as sv
from ultralytics import YOLO
from paddleocr import PaddleOCR
import typer
import numpy as np
import logging

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =========================================================
# INITIALIZE OCR
# =========================================================
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# =========================================================
# LOAD MODELS
# =========================================================
plate_model = YOLO("lpr.pt")     # License Plate YOLO model
helmet_model = YOLO("best.pt")  # Helmet YOLO model

logging.info(f"Helmet classes: {helmet_model.names}")
logging.info(f"Plate classes: {plate_model.names}")

# =========================================================
# CLASS MAPS
# =========================================================
HELMET_LABEL_MAP = {
    0: "With Helmet",
    1: "Without Helmet"
}

PLATE_CLASS_ID = 0  # change if needed after printing plate_model.names

# =========================================================
# OCR FUNCTION
# =========================================================
def perform_ocr(image_rgb):
    if image_rgb is None or image_rgb.size == 0:
        return ""

    try:
        result = ocr.ocr(image_rgb, rec=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                texts.append(line[1][0])
        return "".join(texts)
    except Exception as e:
        logging.error(f"OCR Error: {e}")
        return ""

# =========================================================
# PLATE ↔ NO-HELMET ASSOCIATION (CORRECT LOGIC)
# =========================================================
def belongs_to_no_helmet(plate_box, helmet_box):
    px1, py1, px2, py2 = plate_box
    hx1, hy1, hx2, hy2 = helmet_box

    plate_center_x = (px1 + px2) // 2

    return (
        hy2 < py1 and                  # plate below helmet
        hx1 - 60 < plate_center_x < hx2 + 60
    )

# =========================================================
# VIDEO PROCESSING
# =========================================================
def process_video(video_source, output_file):

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logging.error("❌ Could not open video")
        return

    W, H = 1280, 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (W, H)
    )

    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (W, H))
        annotated = frame.copy()

        # =================================================
        # HELMET DETECTION
        # =================================================
        helmet_results = helmet_model(frame, conf=0.4)[0]
        helmet_dets = sv.Detections.from_ultralytics(helmet_results)

        no_helmet_boxes = []
        helmet_labels = []

        for xyxy, cls_id in zip(helmet_dets.xyxy, helmet_dets.class_id):
            label = HELMET_LABEL_MAP.get(int(cls_id), str(cls_id))
            helmet_labels.append(label)

            if label == "Without Helmet":
                no_helmet_boxes.append(list(map(int, xyxy)))

        annotated = box_annotator.annotate(annotated, helmet_dets)
        annotated = label_annotator.annotate(annotated, helmet_dets, helmet_labels)

        # =================================================
        # LICENSE PLATE DETECTION
        # =================================================
        plate_results = plate_model(frame, conf=0.4)[0]
        plate_dets = sv.Detections.from_ultralytics(plate_results)

        for xyxy, cls_id, conf in zip(
            plate_dets.xyxy,
            plate_dets.class_id,
            plate_dets.confidence
        ):
            if int(cls_id) != PLATE_CLASS_ID:
                continue

            plate_box = list(map(int, xyxy))

            # associate with no-helmet rider
            matched = False
            for nh_box in no_helmet_boxes:
                if belongs_to_no_helmet(plate_box, nh_box):
                    matched = True
                    break

            if not matched:
                continue

            x1, y1, x2, y2 = plate_box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # =================================================
            # OCR PREPROCESSING (IMPORTANT)
            # =================================================
            crop = cv2.resize(crop, (320, 100))
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

            crop_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            plate_text = perform_ocr(crop_rgb)
            plate_text = plate_text.replace(" ", "").replace("-", "")

            if plate_text:
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

        out.write(annotated)
        cv2.imshow("NO HELMET LICENSE PLATE DETECTION", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# =========================================================
# CLI
# =========================================================
app = typer.Typer()

@app.command()
def run(
    video: str = "demo.mp4",
    output: str = "output.avi"
):
    process_video(video, output)

if __name__ == "__main__":
    app()
