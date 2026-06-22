import cv2
import time
from inference_sdk import InferenceHTTPClient

# =========================================================
# ROBOFLOW CLIENT
# =========================================================
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="C8AFB51ehMizNKueKwch"   # 🔴 replace
)

MODEL_ID = "licence-plate-dausq-6wvhb/1"

# =========================================================
# SAFE INFERENCE
# =========================================================
def safe_infer(image, retries=3):
    for i in range(retries):
        try:
            return CLIENT.infer(image, model_id=MODEL_ID)
        except Exception as e:
            print(f"⚠️ Inference failed (retry {i+1}/{retries}):", e)
            time.sleep(1)
    return None

# =========================================================
# VIDEO PROCESSING
# =========================================================
def process_video(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    frame_count = 0
    print("▶ Roboflow Plate Detection Started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🔥 process every 5th frame
        if frame_count % 5 != 0:
            cv2.imshow("Roboflow Plate Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # 🔥 resize to reduce timeout
        frame = cv2.resize(frame, (640, 640))

        # ✅ PASS NUMPY IMAGE (NOT BYTES)
        result = safe_infer(frame)
        if result is None:
            continue

        for pred in result.get("predictions", []):
            x, y = int(pred["x"]), int(pred["y"])
            w, h = int(pred["width"]), int(pred["height"])

            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame, "License Plate",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
            )

        cv2.imshow("Roboflow Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    process_video("demo.mp4")
