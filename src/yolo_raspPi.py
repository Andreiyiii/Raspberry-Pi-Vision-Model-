import cv2
from ultralytics import YOLO
from pathlib import Path

script_dir = Path(__file__).parent
model_path = script_dir.parent / "models" / "best_ncnn_model"
#task='detect' cause it is a NCNN model 
model = YOLO(model_path, task='detect') 
cam = cv2.VideoCapture(0)

#last state of classes to compare and not spam user with information
ultima_banca_vazuta = False
ultimul_gunoi_vazut = False
ultimul_obstacol_vazut = False

while True:
    ret, frame = cam.read()
    if not ret:
        print("Camera not working")
        break
    
    # stream=True for better ram usage
    results = model(frame, stream=True)
    
    #current state of classes
    banca_in_cadrul_curent = False
    gunoi_in_cadrul_curent = False
    obstacol_in_cadrul_curent = False
    
    for r in results: 
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            if cls == 0: # bench 
                banca_in_cadrul_curent = True
            elif cls == 1: # trashcan 
                gunoi_in_cadrul_curent = True
            elif cls == 2: # scooter 
                obstacol_in_cadrul_curent = True

        frame = r.plot()  

    # notify user logic

    if banca_in_cadrul_curent != ultima_banca_vazuta:
        if banca_in_cadrul_curent == True:
            print("Bancă în apropiere. Te poți așeza dacă dorești.")
            
        ultima_banca_vazuta = banca_in_cadrul_curent


    if gunoi_in_cadrul_curent != ultimul_gunoi_vazut:
        if gunoi_in_cadrul_curent == True:
            print("Coș de gunoi în apropiere.")

        ultimul_gunoi_vazut = gunoi_in_cadrul_curent


    if obstacol_in_cadrul_curent != ultimul_obstacol_vazut:
        if obstacol_in_cadrul_curent == True:
            print("Atenție! Obstacol pe traseu.")
            
        ultimul_obstacol_vazut = obstacol_in_cadrul_curent



    cv2.imshow('Raspberry Pi Vision', frame)

    if cv2.waitKey(1) == 27:   # Apasa ESC pentru a iesi
        break

cam.release()
cv2.destroyAllWindows()