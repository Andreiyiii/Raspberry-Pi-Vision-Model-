import math
import requests
from subagenti.locatie import get_locatie

# ce tipuri de locuri putem cauta -> cuvantul cautat in OpenStreetMap
TAGURI = {
    "supermarket": "supermarket",
    "magazin_mic": "convenience store",
    "farmacie": "pharmacy",
    "restaurant": "restaurant",
    "statie_autobuz": "bus stop"
}

# cat de departe cautam: 0.01 grade ~ 1 km in jurul persoanei
MARJA = 0.01


def distanta_metri(lat1, lon1, lat2, lon2):
    # formula haversine - distanta pe glob intre doua puncte
    R = 6371000   # raza Pamantului in metri
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def cauta_loc(tip):
    cuvant = TAGURI.get(tip)
    if cuvant is None:
        return "Nu stiu sa caut tipul: " + tip

    lat, lon = get_locatie()

    # intrebam Nominatim (cautarea publica OpenStreetMap, gratis, fara cheie)
    # viewbox = dreptunghiul in care cautam: stanga, sus, dreapta, jos
    parametri = {
        "q": cuvant,
        "format": "json",
        "limit": 10,
        "viewbox": f"{lon - MARJA},{lat + MARJA},{lon + MARJA},{lat - MARJA}",
        "bounded": 1
    }
    raspuns = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params=parametri,
        headers={"User-Agent": "asistent-nevazatori-proiect-facultate"},   # cerut de Nominatim
        timeout=15
    )
    locuri = raspuns.json()

    if len(locuri) == 0:
        return f"Nu am gasit niciun loc de tip {tip} pe aproape."

    # facem lista (distanta, nume, adresa) si o sortam - cel mai apropiat primul
    lista = []
    for loc in locuri:
        d = distanta_metri(lat, lon, float(loc["lat"]), float(loc["lon"]))
        nume = loc.get("name") or "fara nume"
        adresa = loc["display_name"]
        lista.append((d, nume, adresa))
    lista.sort()

    text = ""
    for d, nume, adresa in lista[:3]:
        text += f"{nume}: {int(d)} metri ({adresa})\n"
    return text


if __name__ == "__main__":
    # test: din folderul asistent ruleaza  python -m subagenti.gps
    print(cauta_loc("supermarket"))
