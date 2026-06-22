# Helmet Detection System

AI-powered Helmet, No-Helmet & License Plate Detection with OCR and Reports.

## Features

- Detect helmets and no-helmets using YOLO
- License plate detection and OCR text extraction
- Associate no-helmet riders with license plates
- Generate CSV and PDF reports
- Beautiful web interface with upload and live camera

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run migrations:
   ```
   python manage.py makemigrations
   python manage.py migrate
   ```

3. Start the server:
   ```
   python manage.py runserver
   ```

4. Access at http://127.0.0.1:8000

## Models

- `best.pt`: Helmet detection model
- `lpr.pt`: License plate detection model

Place these files in the project root.

## Usage

- Upload a video for processing
- Use live camera for real-time detection
- View and download reports
