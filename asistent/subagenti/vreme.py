import requests
from subagenti.locatie import get_locatie


def prognoza(zile):
    # luam prognoza de la Open-Meteo (gratis, fara cheie)
    if zile < 1:
        zile = 1
    if zile > 7:
        zile = 7

    lat, lon = get_locatie()
    parametri = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": zile
    }
    raspuns = requests.get("https://api.open-meteo.com/v1/forecast", params=parametri, timeout=20)
    date = raspuns.json()["daily"]

    text = ""
    for i in range(len(date["time"])):
        zi = date["time"][i]
        maxim = date["temperature_2m_max"][i]
        minim = date["temperature_2m_min"][i]
        ploaie = date["precipitation_probability_max"][i]
        text += f"{zi}: maxim {maxim} grade, minim {minim} grade, sansa de ploaie {ploaie}%\n"
    return text


if __name__ == "__main__":
    # test: din folderul asistent ruleaza  python -m subagenti.vreme
    print(prognoza(3))
