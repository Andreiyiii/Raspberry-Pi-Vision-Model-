import datetime
import anthropic
import config
from subagenti.gps import cauta_loc, TAGURI
from subagenti.vreme import prognoza
from subagenti.imagine import intreaba_imaginea

client = anthropic.Anthropic()   # cheia se ia din variabila ANTHROPIC_API_KEY

INSTRUCTIUNI = (
    "Ajuti o persoana nevazatoare care a apasat un buton si a pus o intrebare. "
    "Camera ei a facut o poza chiar acum, dar o poti vedea doar prin unealta intreaba_imaginea. "
    "Foloseste uneltele cand ai nevoie de informatii. "
    "Raspunsul va fi citit cu voce tare: scrie in romana, 1-2 propozitii, fara liste, fara emoji, fara formatare."
)

# uneltele (subagentii) pe care orchestratorul le poate folosi
UNELTE = [
    {
        "name": "cauta_loc",
        "description": "Cauta cele mai apropiate locuri de un anumit tip langa persoana (raza 1 km). "
                       "Intoarce numele si distanta in metri.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "tip": {"type": "string", "enum": list(TAGURI.keys())}   # tipurile din gps.py
            },
            "required": ["tip"],
            "additionalProperties": False
        }
    },
    {
        "name": "prognoza_vreme",
        "description": "Prognoza meteo pentru locatia persoanei: temperatura maxima/minima si sansa de ploaie "
                       "pentru fiecare zi, incepand de azi.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "zile": {"type": "integer", "description": "cate zile, intre 1 si 7 (1 = doar azi)"}
            },
            "required": ["zile"],
            "additionalProperties": False
        }
    },
    {
        "name": "intreaba_imaginea",
        "description": "Pune o intrebare despre poza facuta de camera persoanei "
                       "(ex: unde e telecomanda, ce scrie pe eticheta, ce culoare are bluza).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "intrebare": {"type": "string", "description": "intrebarea despre poza"}
            },
            "required": ["intrebare"],
            "additionalProperties": False
        }
    }
]


def ruleaza_subagent(nume, date):
    # alegem subagentul dupa numele uneltei cerute de Claude
    try:
        if nume == "cauta_loc":
            return cauta_loc(date["tip"])
        elif nume == "prognoza_vreme":
            return prognoza(date["zile"])
        elif nume == "intreaba_imaginea":
            return intreaba_imaginea(date["intrebare"])
        else:
            return "Unealta necunoscuta: " + nume
    except Exception as e:
        # ii spunem lui Claude ca subagentul a dat eroare, ca sa poata raspunde ceva
        return "Eroare la subagent: " + str(e)


def raspunde(intrebare):
    azi = datetime.date.today().isoformat()
    mesaje = [{"role": "user", "content": f"Data de azi: {azi}\nIntrebarea: {intrebare}"}]

    # maxim 5 runde de "Claude cere un subagent -> ii dam raspunsul"
    for runda in range(5):
        raspuns = client.beta.messages.create(
            model=config.MODEL,
            max_tokens=4000,
            output_config={"effort": "low"},   # raspuns rapid, persoana asteapta
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",               # daca modelul refuza, incearca automat alt model
            system=INSTRUCTIUNI,
            tools=UNELTE,
            messages=mesaje
        )

        # daca nu mai cere niciun subagent, avem raspunsul final
        if raspuns.stop_reason != "tool_use":
            break

        # pastram tot raspunsul lui Claude in conversatie
        mesaje.append({"role": "assistant", "content": raspuns.content})

        # rulam fiecare subagent cerut si trimitem rezultatele inapoi, toate odata
        rezultate = []
        for bloc in raspuns.content:
            if bloc.type == "tool_use":
                print("Subagent:", bloc.name, bloc.input)
                rezultat = ruleaza_subagent(bloc.name, bloc.input)
                rezultate.append({"type": "tool_result", "tool_use_id": bloc.id, "content": rezultat})
        mesaje.append({"role": "user", "content": rezultate})

    for bloc in raspuns.content:
        if bloc.type == "text":
            return bloc.text
    return "Imi pare rau, nu am putut raspunde la intrebare."


if __name__ == "__main__":
    # test fara microfon: din folderul asistent ruleaza  python orchestrator.py
    print(raspunde("Va ploua maine?"))
