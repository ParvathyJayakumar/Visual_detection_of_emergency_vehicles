from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Use your model (e.g., yolov8n, yolov8s, etc.)

# Train the model
model.train(data="C:\\Program\\dataset\\data.yaml", epochs=50, imgsz=640, batch=16)
