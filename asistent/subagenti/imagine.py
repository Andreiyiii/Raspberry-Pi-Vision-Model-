import base64
import anthropic
import config

client = anthropic.Anthropic()   # cheia se ia din variabila ANTHROPIC_API_KEY

INSTRUCTIUNI = (
    "Ajuti o persoana nevazatoare. Raspunde in romana, scurt, la intrebarea despre poza. "
    "Spune pozitia obiectelor ca: in stanga / in centru / in dreapta, aproape / departe, "
    "sus / jos. Daca nu vezi obiectul, spune clar asta."
)


def intreaba_imaginea(intrebare):
    # citim poza facuta la apasarea butonului si o trimitem la Claude
    with open(config.CALE_POZA, "rb") as f:
        poza = base64.standard_b64encode(f.read()).decode("utf-8")

    raspuns = client.messages.create(
        model=config.MODEL,
        max_tokens=2000,
        output_config={"effort": "low"},   # raspuns rapid
        system=INSTRUCTIUNI,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": poza}},
                {"type": "text", "text": intrebare}
            ]
        }]
    )

    for bloc in raspuns.content:
        if bloc.type == "text":
            return bloc.text
    return "Nu am putut analiza poza."


if __name__ == "__main__":
    # test: pune o poza poza.jpg in folderul asistent si ruleaza  python -m subagenti.imagine
    print(intreaba_imaginea("Ce se vede in poza?"))
