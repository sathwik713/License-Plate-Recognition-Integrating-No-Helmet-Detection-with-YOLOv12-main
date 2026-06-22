from paddleocr import PaddleOCR
import cv2
import logging

logging.basicConfig(level=logging.INFO)

ocr = PaddleOCR(use_angle_cls=True, lang="en")

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

def preprocess_plate(crop):
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
    return crop_rgb