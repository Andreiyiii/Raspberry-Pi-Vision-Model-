import cv2
from ultralytics import YOLO
from pathlib import Path

script_dir = Path(__file__).parent
model_path = script_dir.parent / "models" / "task_ncnn_model"
#task='detect' cause it is a NCNN model 
model = YOLO(model_path, task='detect') 

cam = cv2.VideoCapture(0)

#last state of classes to compare and not spam user with information
ultimul_semafor_vazut = None
ultima_stare_trecere = False
ultima_banca_vazuta = False
ultimul_gunoi_vazut = False
ultimul_obstacol_vazut = False

while True:
    ret, frame = cam.read()
    if not ret:
        print("Eroare la citirea camerei")
        break
    
    # stream=True for better ram usage
    results = model(frame, stream=True)
    
    #current state of classes
    semafor_in_cadrul_curent = None
    trecere_in_cadrul_curent = False
    banca_in_cadrul_curent = False
    gunoi_in_cadrul_curent = False
    obstacol_in_cadrul_curent = False
    
    for r in results: 
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            if cls == 0: #crosswalk
                trecere_in_cadrul_curent = True
            elif cls == 1: 
                semafor_in_cadrul_curent = "ROSU"
            elif cls == 2: 
                semafor_in_cadrul_curent = "VERDE"
            elif cls == 4: # bench 
                banca_in_cadrul_curent = True
            elif cls == 5: # trashcan 
                gunoi_in_cadrul_curent = True
            elif cls == 6: # scooter 
                obstacol_in_cadrul_curent = True

        frame = r.plot()  

    # notify user logic


    cv2.imshow('Raspberry Pi Vision', frame)

    if cv2.waitKey(1) == 27:   # Apasa ESC pentru a iesi
        break

cam.release()
cv2.destroyAllWindows()