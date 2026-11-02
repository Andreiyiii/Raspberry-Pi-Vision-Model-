import config


def get_locatie():
    # deocamdata locatia fixa din config.py
    # mai tarziu aici citim GPS-ul trimis de telefon (sau de un modul GPS)
    return config.LATITUDINE, config.LONGITUDINE


if __name__ == "__main__":
    # test: din folderul asistent ruleaza  python -m subagenti.locatie
    print(get_locatie())
