from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.conf import settings
from django.utils import timezone
import cv2
import os
from .ai.pipeline import process_frame
from .models import Detection
import csv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import cv2
import supervision as sv
from ultralytics import YOLO
from paddleocr import PaddleOCR
import typer
import numpy as np
import logging


def home(request):
    return render(request, 'home.html')

def upload_video(request):
    return render(request, 'upload.html')

def process_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_file.name)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        return JsonResponse({'message': 'Video uploaded, starting live detection', 'stream_url': f'/stream/{video_file.name}'})

    return JsonResponse({'error': 'Invalid request'})

def live_camera(request):
    return render(request, 'live.html')

def report(request):
    detections = Detection.objects.all()
    return render(request, 'report.html', {'detections': detections})

def training_results(request):
    import pandas as pd
    results1=pd.read_csv(r'results.csv')
    dff = results1.to_html(classes='table table-striped', index=False) 
    return render(request, 'training.html', {'data': dff})

def generate_report(request):
    format_type = request.GET.get('format', 'csv')
    detections = Detection.objects.all()

    if format_type == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="report.csv"'
        writer = csv.writer(response)
        writer.writerow(['Date', 'Time', 'Helmet Status', 'License Plate', 'Video Source'])
        for det in detections:
            writer.writerow([det.date, det.time, det.helmet_status, det.license_plate_text, det.video_source])
        return response

    elif format_type == 'pdf':
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.drawString(100, 750, "Helmet Detection Report")
        y = 700
        for det in detections:
            p.drawString(100, y, f"{det.date} {det.time} - {det.helmet_status} - {det.license_plate_text}")
            y -= 20
            if y < 50:
                p.showPage()
                y = 750
        p.save()
        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="report.pdf"'
        return response

    return HttpResponse("Invalid format")

def video_detection_stream(video_path):
    from .ai.helmet_detector import detect_helmets
    from .ai.plate_detector import detect_plates
    from .ai.ocr import perform_ocr, preprocess_plate

    cap = cv2.VideoCapture(video_path)
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.8)

    def belongs_to_no_helmet(plate_box, helmet_box):
        px1, py1, px2, py2 = plate_box
        hx1, hy1, hx2, hy2 = helmet_box
        plate_center_x = (px1 + px2) // 2
        return hy2 < py1 and hx1 - 60 < plate_center_x < hx2 + 60

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        # Helmet detection
        no_helmet_boxes, helmet_dets, helmet_labels = detect_helmets(frame)
        annotated = box_annotator.annotate(annotated, helmet_dets)
        annotated = label_annotator.annotate(annotated, helmet_dets, helmet_labels)

        # Plate detection
        plates, plate_dets = detect_plates(frame)

        for plate_box in plates:
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

            crop_rgb = preprocess_plate(crop)
            plate_text = perform_ocr(crop_rgb)
            plate_text = plate_text.replace(" ", "").replace("-", "")

            if plate_text:
                # Generate challan PDF
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                p.drawString(100, 750, "Traffic Violation Challan")
                p.drawString(100, 720, f"Date: {timezone.now().date()}")
                p.drawString(100, 690, f"Time: {timezone.now().time()}")
                p.drawString(100, 660, f"Violation: No Helmet")
                p.drawString(100, 630, f"License Plate: {plate_text}")
                p.drawString(100, 600, f"Fine Amount: Rs. 500")
                p.drawString(100, 570, f"Video Source: {os.path.basename(video_path)}")
                p.save()
                buffer.seek(0)

                challan_filename = f"challan_{plate_text}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                challan_path = os.path.join(settings.MEDIA_ROOT, 'challans', challan_filename)
                os.makedirs(os.path.dirname(challan_path), exist_ok=True)
                with open(challan_path, 'wb') as f:
                    f.write(buffer.getvalue())

                Detection.objects.create(
                    helmet_status='No Helmet',
                    license_plate_text=plate_text,
                    video_source=os.path.basename(video_path),
                    fine_amount=500,
                    challan_pdf=f'challans/{challan_filename}'
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"NO HELMET: {plate_text} - FINE: Rs.500",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

        _, jpeg = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

def stream_uploaded_video(request, filename):
    video_path = os.path.join(settings.MEDIA_ROOT, 'videos', filename)
    return StreamingHttpResponse(
        video_detection_stream(video_path),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )