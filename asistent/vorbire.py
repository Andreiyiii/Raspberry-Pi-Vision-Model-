import pyttsx3

motor = pyttsx3.init()

# cautam o voce romaneasca, daca exista pe sistem
for voce in motor.getProperty("voices"):
    if "roman" in voce.name.lower():
        motor.setProperty("voice", voce.id)
        break


def spune(text):
    print("Raspuns:", text)
    motor.say(text)
    motor.runAndWait()
