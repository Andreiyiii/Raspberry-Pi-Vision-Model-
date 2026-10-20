# setari pentru asistentul vocal - schimba aici, nu prin cod
from pathlib import Path

MODEL = "claude-opus-5-5"

# locatia fixa pana punem un modul GPS (acum: Piata Unirii, Bucuresti)
LATITUDINE = 44.4268
LONGITUDINE = 26.1025

LIMBA = "ro-RO"          # limba pentru recunoasterea vocii

PE_RASPBERRY = False     # False = test pe laptop (Enter + webcam), True = buton + camera Pi
PIN_BUTON = 17           # pinul GPIO la care e legat butonul

# unde salvam poza facuta la apasarea butonului (mereu in folderul asistent,
# indiferent din ce folder pornim programul)
CALE_POZA = str(Path(__file__).parent / "poza.jpg")
