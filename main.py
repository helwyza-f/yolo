from ultralytics import YOLO
import torch
import os

def train_model():
    if torch.cuda.is_available():
        print("✅ GPU terdeteksi:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ GPU tidak tersedia, training pakai CPU")

    model = YOLO("yolov8n.pt")  # base model
    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0  # gunakan GPU
    )

def test_model(image_path="test.jpg"):
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    model = YOLO(model_path)
    print(f"🔍 Menguji gambar: {image_path}")
    results = model.predict(source=image_path, save=True)

    for result in results:
        boxes = result.boxes
        print(f"📦 Deteksi: {len(boxes)} objek")
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"- {model.names[cls]} ({conf:.2f})")

if __name__ == "__main__":
    # Uncomment salah satu di bawah sesuai kebutuhan:

    # Untuk training:
    # train_model()

    # Untuk testing gambar:
    test_model("test2.jpeg")  # Ganti "test.jpg" sesuai path file yang ingin diuji
