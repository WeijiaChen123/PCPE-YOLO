from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model

model = YOLO("F:/yolo_data/experiment/(7-3)YOLO12l-Visdrone/weights/best.pt")  # load a custom model
# Predict with the model
results = model.predict("C:/Users/Lenovo/Desktop/img1.png",save=True,hide_conf=True)  # predict on an image


