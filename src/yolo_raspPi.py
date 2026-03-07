import cv2
from ultralytics import YOLO
from pathlib import Path
from picamera2 import Picamera2 

script_dir = Path(__file__).parent
model_path = script_dir.parent / "models" / "best_ncnn_model"
model = YOLO(model_path, task='detect')

# connecting to native rasppi camera since opencv didnt work
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

print("Camera Pi 5 a pornit cu succes!")

ultima_banca_vazuta = False
ultimul_gunoi_vazut = False
ultimul_obstacol_vazut = False

while True:
    frame = picam2.capture_array()
    # Camera trimite culorile in format RGB, le intoarcem in BGR pentru YOLO
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # stream=True for better ram usage
    results = model(frame, stream=True)
    
    banca_in_cadrul_curent = False
    gunoi_in_cadrul_curent = False
    obstacol_in_cadrul_curent = False
    
    for r in results: 
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            if cls == 0: 
                banca_in_cadrul_curent = True
            elif cls == 1: 
                gunoi_in_cadrul_curent = True
            elif cls == 2: 
                obstacol_in_cadrul_curent = True

        frame = r.plot()  

    if banca_in_cadrul_curent != ultima_banca_vazuta:
        if banca_in_cadrul_curent == True:
            print("Banca in apropiere. Te poti aseza daca doresti.")
        ultima_banca_vazuta = banca_in_cadrul_curent

    if gunoi_in_cadrul_curent != ultimul_gunoi_vazut:
        if gunoi_in_cadrul_curent == True:
            print("Cos de gunoi in apropiere.")
        ultimul_gunoi_vazut = gunoi_in_cadrul_curent

    if obstacol_in_cadrul_curent != ultimul_obstacol_vazut:
        if obstacol_in_cadrul_curent == True:
            print("Atentie! Obstacol pe traseu.")
        ultimul_obstacol_vazut = obstacol_in_cadrul_curent

    cv2.imshow('Raspberry Pi Vision', frame)

    if cv2.waitKey(1) == 27:   # Apasa ESC pentru a iesi
        break

picam2.stop()
cv2.destroyAllWindows()