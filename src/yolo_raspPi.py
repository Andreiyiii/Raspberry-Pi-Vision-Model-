import cv2
import sys
import time
import threading
from ultralytics import YOLO
from pathlib import Path
from analyze import analizeaza
from voice import talk
from picamera2 import Picamera2 #type: ignore 
from libcamera import controls  #type: ignore
                                #+ changes in venv/pyvenv.cfg  include-system-site-packages = true
from gpiozero import Button     #type: ignore


script_dir = Path(__file__).parent

# asistentul vocal e in folderul asistent/, il adaugam ca sa-l putem importa
sys.path.append(str(script_dir.parent / "asistent"))
from main import intreaba_asistentul
from config import PIN_BUTON, CALE_POZA   # din asistent/config.py

buton = Button(PIN_BUTON)
buton_apasat = False


def la_apasare():
    # gpiozero apeleaza functia asta singur cand se apasa butonul,
    # asa nu pierdem o apasare scurta intre doua frame-uri
    global buton_apasat
    buton_apasat = True


buton.when_pressed = la_apasare
model_path = script_dir.parent / "models" / "best_ncnn_model"
model = YOLO(model_path, task='detect')

# connecting to native rasppi camera since opencv didnt work
# Camera Module 3 Wide (IMX708, 16:9, ~120°) - folosim 16:9 ca sa nu pierdem din unghi
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 360), "format": "RGB888"},  # RGB888 in picamera2 = BGR in memorie, direct pt OpenCV
    sensor={"output_size": (2304, 1296)},           # mod binned cu tot campul vizual
)
picam2.configure(config)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})  # autofocus continuu
picam2.start()

print("Camera Pi 5 a pornit cu succes!")

anuntate = set()     # obiectele deja anuntate care inca sunt in fata camerei

# ca sa nu vorbeasca prea des
PAUZA = 2.0          # secunde de asteptare intre doua anunturi
ultimul_anunt = 0    # cand am vorbit ultima data
vorbeste_acum = False
asistent_activ = False   # True cat timp persoana vorbeste cu asistentul


def vorbeste(cheie):
    # rulam talk() intr-un thread ca sa nu blocheze camera cat vorbeste
    global vorbeste_acum
    vorbeste_acum = True
    talk(cheie)
    vorbeste_acum = False


def porneste_asistentul(poza):
    # rulam asistentul intr-un thread ca detectia sa mearga mai departe
    global asistent_activ

    # asteptam sa termine anuntul de obstacol, ca sa nu vorbeasca amandoua deodata
    while vorbeste_acum:
        time.sleep(0.1)

    try:
        cv2.imwrite(CALE_POZA, poza)
        intreaba_asistentul()
    except Exception as e:
        print("Eroare asistent:", e)

    # orice s-ar intampla, pornim inapoi anunturile de obstacole
    asistent_activ = False


while True:
    frame = picam2.capture_array()  # deja BGR (format RGB888)

    # butonul apasat -> trimitem poza curenta la asistent (camera e deja pornita aici)
    if buton_apasat and not asistent_activ:
        buton_apasat = False
        asistent_activ = True   # il punem aici, ca sa nu porneasca de 2 ori daca butonul e tinut apasat
        threading.Thread(target=porneste_asistentul, args=(frame.copy(),)).start()
    
    # conf=0.5 ca sa ignore detectiile nesigure, verbose=False ca sa nu scrie in consola
    # imgsz=512 = marimea la care a fost exportat modelul
    results = model.predict(frame, imgsz=512, conf=0.5, verbose=False)

    for r in results:
        rezultate = analizeaza(r.boxes, r.names)

        # mai intai punem TOATE obiectele in set
        obiecte_curente = set()
        for cheie, prioritate in rezultate:
            obiecte_curente.add(cheie)

        # uitam obiectele care au disparut, ca sa fie anuntate din nou daca revin
        ramase = set()
        for cheie in anuntate:
            if cheie in obiecte_curente:
                ramase.add(cheie)
        anuntate = ramase

        # anuntam primul obiect (cel mai periculos) care n-a fost anuntat inca
        # il trecem la "anuntate" DOAR cand chiar vorbim, ca sa nu se piarda in pauza
        # nu anuntam obstacole cat timp vorbeste asistentul, ca sa nu se suprapuna vocile
        if not vorbeste_acum and not asistent_activ and time.time() - ultimul_anunt > PAUZA:
            for cheie, prioritate in rezultate:
                if cheie not in anuntate:
                    threading.Thread(target=vorbeste, args=(cheie,)).start()
                    ultimul_anunt = time.time()
                    anuntate.add(cheie)
                    break

        frame = r.plot()

    cv2.imshow('Raspberry Pi Vision', frame)

    if cv2.waitKey(1) == 27:   # Apasa ESC pentru a iesi
        break

picam2.stop()
cv2.destroyAllWindows()