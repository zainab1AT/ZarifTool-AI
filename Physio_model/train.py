from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')  # Load a pretrained model
model.train(data='C:/projectstask_physio/config.yaml', epochs=1, imgsz=640)
