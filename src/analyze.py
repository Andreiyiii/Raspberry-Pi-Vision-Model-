# numele din model -> numele scurt folosit la mp3
# mergem dupa nume, nu dupa numar, pentru ca Roboflow schimba numerele cand adaugi clase
NUME_SCURT = {
    "bench": "bench",
    "concrete_post": "stalp",
    "electric scooter": "scooter",
    "human": "human",
    "pillar": "stalp",
    "plastic_post": "stalp",
    "trashcan": "trashcan"
}

# cat din inaltimea imaginii (0..1) trebuie sa ocupe obiectul ca sa fie "aproape"
# obiectele joase (banca, cos) au prag mai mic, cele inalte (stalp, om) mai mare
PRAG_APROAPE = {
    "bench": 0.25,
    "stalp": 0.5,
    "scooter": 0.3,
    "trashcan": 0.3,
    "human": 0.5
}

def get_directie(box):
    x1, y1, x2, y2 = box.xyxyn[0]   # coordonate normalizate 0..1
    x_centru = (x1 + x2) / 2
    
    if x_centru < 0.33:
        return "stanga"
    elif x_centru > 0.66:
        return "dreapta"
    else:
        return "fata"

def get_distanta(box, nume):
    x1, y1, x2, y2 = box.xyxyn[0]
    procent = y2 - y1               # deja procent din inaltimea imaginii
    
    if procent > PRAG_APROAPE[nume]:
        return "aproape"
    else:
        return "departe"

def get_prioritate(distanta, directie):
    # 3 = aproape si in fata (cel mai periculos), 2 = aproape lateral, 1 = departe
    if distanta == "aproape" and directie == "fata":
        return 3
    elif distanta == "aproape":
        return 2
    else:
        return 1

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

def analizeaza(boxes, names):
    rezultate = []
    numar_oameni = 0
    oameni_boxes = []

    for box in boxes:
        cls = int(box.cls[0])
        nume = NUME_SCURT.get(names[cls])   # names vine din model, ex: {0: "bench", ...}
        if nume is None:
            continue
            
        if nume == "human":
            numar_oameni += 1
            oameni_boxes.append(box)
            continue
        
        distanta = get_distanta(box, nume)
        directie = get_directie(box)
        cheie = f"{nume}_{distanta}_{directie}"
        rezultate.append((cheie, get_prioritate(distanta, directie)))
    
    # logica oameni
    if numar_oameni >= 3:
        rezultate.append(("human_aglomerat", 1))
    elif numar_oameni > 0:
        cel_mai_apropiat = get_cel_mai_apropiat_om(oameni_boxes)
        distanta = get_distanta(cel_mai_apropiat, "human")
        directie = get_directie(cel_mai_apropiat)
        rezultate.append((f"human_{distanta}_{directie}", get_prioritate(distanta, directie)))
    
    # sorteaza dupa prioritate - cel mai periculos primul
    rezultate.sort(key=lambda x: x[1], reverse=True)
    return rezultate
