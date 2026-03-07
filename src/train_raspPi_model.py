from ultralytics import YOLO

model = YOLO('models/yolov8n.pt') 


results = model.train(
    data='data/data.yaml', 
    epochs=100,        
    imgsz=512,         
    degrees=15.0,      
    mosaic=1.0,        
    mixup=0.2,         
    hsv_s=0.5,         
    fliplr=0.5         
)
best_model = YOLO('runs/detect/train/weights/best.pt')
best_model.export(format='ncnn')