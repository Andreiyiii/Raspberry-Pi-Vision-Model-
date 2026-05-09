FRAME_W = 640
FRAME_H = 480

CLASS_NAMES = {
    0: "bench",
    1: "stalp",
    2: "scooter",
    3: "stalp",
    4: "trashcan",
    5: "human"
}

def get_directie(box):
    x1, y1, x2, y2 = box.xyxy[0]
    x_centru = (x1 + x2) / 2
    
    if x_centru < FRAME_W * 0.33:
        return "stanga"
    elif x_centru > FRAME_W * 0.66:
        return "dreapta"
    else:
        return "fata"

def get_distanta(box):
    x1, y1, x2, y2 = box.xyxy[0]
    inaltime_box = y2 - y1
    procent = inaltime_box / FRAME_H
    
    if procent > 0.4:
        return "aproape"
    else:
        return "departe"

def get_cel_mai_apropiat_om(oameni_boxes):
    cel_mai_apropiat = None
    inaltime_maxima = 0
    
    for box in oameni_boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        inaltime = y2 - y1
        
        if inaltime > inaltime_maxima:
            inaltime_maxima = inaltime
            cel_mai_apropiat = box
    
    return cel_mai_apropiat

def analizeaza(boxes):
    rezultate = []
    numar_oameni = 0
    oameni_boxes = []

    for box in boxes:
        cls = int(box.cls[0])
        nume = CLASS_NAMES.get(cls)
        if nume is None:
            continue
            
        if cls == 5:
            numar_oameni += 1
            oameni_boxes.append(box)
            continue
        
        distanta = get_distanta(box)
        directie = get_directie(box)
        cheie = f"{nume}_{distanta}_{directie}"
        prioritate = 2 if distanta == "aproape" else 1
        rezultate.append((cheie, prioritate))
    
    # logica oameni
    if numar_oameni >= 3:
        rezultate.append(("human_aglomerat", 1))
    elif numar_oameni > 0:
        cel_mai_apropiat = get_cel_mai_apropiat_om(oameni_boxes)
        distanta = get_distanta(cel_mai_apropiat)
        directie = get_directie(cel_mai_apropiat)
        prioritate = 2 if distanta == "aproape" else 1
        rezultate.append((f"human_{distanta}_{directie}", prioritate))
    
    # sorteaza dupa prioritate - cel mai periculos primul
    rezultate.sort(key=lambda x: x[1], reverse=True)
    return rezultate