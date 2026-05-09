import cv2
from ultralytics import YOLO
from pathlib import Path  
from analyze import analizeaza
from voice import talk
from picamera2 import Picamera2 #type: ignore 
                                #+ changes in venv/pyvenv.cfg  include-system-site-packages = true


script_dir = Path(__file__).parent
model_path = script_dir.parent / "models" / "best_ncnn_model"
model = YOLO(model_path, task='detect')

# connecting to native rasppi camera since opencv didnt work
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

print("Camera Pi 5 a pornit cu succes!")

obiecte_anterioare = set()


while True:
    frame = picam2.capture_array()
    # camera sends RGB, we switch to BGR for YOLO
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # stream=True for better ram usage
    results = model(frame, stream=True)
    
    for r in results:
        rezultate = analizeaza(r.boxes)

        obiecte_curente = set()
        for cheie, prioritate in rezultate:
            obiecte_curente.add(cheie)
            
            if cheie not in obiecte_anterioare:
                talk(cheie)
                break
        
        obiecte_anterioare = obiecte_curente
        frame = r.plot()  

    cv2.imshow('Raspberry Pi Vision', frame)

    if cv2.waitKey(1) == 27:   # Apasa ESC pentru a iesi
        break

picam2.stop()
cv2.destroyAllWindows()