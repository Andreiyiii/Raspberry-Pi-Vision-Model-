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
# results.save_dir e folderul in care a salvat chiar acum (train, train2, train3...)
# asa nu mai trebuie sa scriem noi numele folderului de mana
best_path = str(results.save_dir) + "/weights/best.pt"
print("Modelul bun este la:", best_path)

best_model = YOLO(best_path)
best_model.export(format='ncnn', imgsz=512)