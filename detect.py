from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('runs/detect/train4/weights/best.pt')  # Path to your trained model

# Predict using the webcam and filter only the "Ambulance" class (class index 0)
results = model.predict(source=0, classes=[0], show=True)  # Replace 0 with the correct index for "Ambulance"
