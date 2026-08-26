import cv2
import time
import threading
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

# ca sa nu vorbeasca prea des
PAUZA = 2.0          # secunde de asteptare intre doua anunturi
ultimul_anunt = 0    # cand am vorbit ultima data
vorbeste_acum = False


def vorbeste(cheie):
    # rulam talk() intr-un thread ca sa nu blocheze camera cat vorbeste
    global vorbeste_acum
    vorbeste_acum = True
    talk(cheie)
    vorbeste_acum = False


while True:
    frame = picam2.capture_array()
    # camera sends RGB, we switch to BGR for YOLO
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # conf=0.5 ca sa ignore detectiile nesigure, verbose=False ca sa nu scrie in consola
    results = model.predict(frame, conf=0.5, verbose=False)

    for r in results:
        rezultate = analizeaza(r.boxes)

        # mai intai punem TOATE obiectele in set
        obiecte_curente = set()
        for cheie, prioritate in rezultate:
            obiecte_curente.add(cheie)

        # abia apoi vorbim, ca sa nu repete la infinit
        for cheie, prioritate in rezultate:
            if cheie not in obiecte_anterioare:
                # vorbim doar daca nu vorbeste deja si a trecut PAUZA
                if not vorbeste_acum and time.time() - ultimul_anunt > PAUZA:
                    threading.Thread(target=vorbeste, args=(cheie,)).start()
                    ultimul_anunt = time.time()
                break

        obiecte_anterioare = obiecte_curente
        frame = r.plot()

    cv2.imshow('Raspberry Pi Vision', frame)

    if cv2.waitKey(1) == 27:   # Apasa ESC pentru a iesi
        break

picam2.stop()
cv2.destroyAllWindows()