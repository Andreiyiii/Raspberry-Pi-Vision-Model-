from ultralytics import YOLO

model = YOLO('models/yolov8n.pt') 


best_model = YOLO('runs/detect/train2/weights/best.pt')
best_model.export(format='ncnn', imgsz=512)