import cv2
import config
from microfon import asculta
from vorbire import spune
from orchestrator import raspunde

def asteapta_butonul():
    if config.PE_RASPBERRY:
        buton.wait_for_press()
    else:
        input("Apasa Enter ca sa pui o intrebare...")


def fa_poza():
    if config.PE_RASPBERRY:
        # atentie: merge doar daca src/yolo_raspPi.py NU ruleaza (camera e folosita de un singur program)
        from picamera2 import Picamera2  # type: ignore
        camera = Picamera2()
        camera.configure(camera.create_still_configuration())
        camera.start()
        camera.capture_file(config.CALE_POZA)
        camera.stop()
        camera.close()
    else:
        # pe laptop folosim webcam-ul
        cam = cv2.VideoCapture(0)
        ok, frame = cam.read()
        cam.release()
        if ok:
            cv2.imwrite(config.CALE_POZA, frame)
        else:
            print("Nu am putut face poza cu webcam-ul")


def intreaba_asistentul():
    # poza e deja salvata in config.CALE_POZA; acum ascultam intrebarea si raspundem
    # (folosita si de src/yolo_raspPi.py cand se apasa butonul)
    spune("Te ascult")   # semnal ca poate vorbi

    intrebare = asculta()
    if intrebare == "":
        spune("Nu am inteles, incearca din nou.")
        return
    print("Intrebare:", intrebare)

    try:
        spune(raspunde(intrebare))
    except Exception as e:
        print("Eroare:", e)
        spune("Am o problema cu internetul, incearca din nou.")


# partea de mai jos ruleaza doar cand pornim direct  python main.py
# (nu si cand src/yolo_raspPi.py importa intreaba_asistentul)
if __name__ == "__main__":
    if config.PE_RASPBERRY:
        from gpiozero import Button  # type: ignore
        buton = Button(config.PIN_BUTON)

    while True:
        asteapta_butonul()
        fa_poza()
        intreaba_asistentul()
