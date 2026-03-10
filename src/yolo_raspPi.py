import cv2
from ultralytics import YOLO
from pathlib import Path  
from picamera2 import Picamera2 #type: ignore 
                                #+ changes in venv/pyvenv.cfg  include-system-site-packages = true


def notify(stare_curenta, stare_anterioara, mesaj):
    if stare_curenta != stare_anterioara:
        if stare_curenta == True:
            print(mesaj)
    
    return stare_curenta



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
ultimul_stalp_vazut = False
ultima_trotineta_vazuta = False

while True:
    frame = picam2.capture_array()
    # Camera trimite culorile in format RGB, le intoarcem in BGR pentru YOLO
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # stream=True for better ram usage
    results = model(frame, stream=True)
    
    banca_in_cadrul_curent = False
    stalp_in_cadrul_curent = False
    trotineta_in_cadrul_curent = False
    gunoi_in_cadrul_curent = False
    
    for r in results: 
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            if cls == 0: # bench
                banca_in_cadrul_curent = True
            elif cls in [1, 3]: # concrete_post sau plastic_post 
                stalp_in_cadrul_curent = True
            elif cls == 2: # electric scooter
                trotineta_in_cadrul_curent = True
            elif cls == 4: # trashcan
                gunoi_in_cadrul_curent = True

        frame = r.plot()  

    ultima_banca_vazuta = notify(banca_in_cadrul_curent, ultima_banca_vazuta, "banca")
    ultimul_stalp_vazut = notify(stalp_in_cadrul_curent, ultimul_stalp_vazut, "stalp")
    ultima_trotineta_vazuta = notify(trotineta_in_cadrul_curent, ultima_trotineta_vazuta, "trotineta")
    ultimul_gunoi_vazut = notify(gunoi_in_cadrul_curent, ultimul_gunoi_vazut, "cos de gunoi")

    cv2.imshow('Raspberry Pi Vision', frame)

    if cv2.waitKey(1) == 27:   # Apasa ESC pentru a iesi
        break

picam2.stop()
cv2.destroyAllWindows()