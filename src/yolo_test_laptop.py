import cv2
from ultralytics import YOLO
from pathlib import Path

script_dir = Path(__file__).parent
model_path = script_dir.parent / "models" / "yolov8n.pt"
model = YOLO(model_path) 

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error, no return from camera")
        break


    results = model(frame, stream=True)
    for r in results: 
        frame = r.plot()  

    cv2.imshow('Testing camera', frame)

    if cv2.waitKey(1) == 27:   #press esc to close
        break

cam.release()
cv2.destroyAllWindows()