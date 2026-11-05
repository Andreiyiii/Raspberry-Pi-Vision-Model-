import speech_recognition as sr
import config


def asculta():
    # inregistreaza intrebarea de la microfon si o transforma in text
    r = sr.Recognizer()

    with sr.Microphone() as sursa:
        r.adjust_for_ambient_noise(sursa, duration=0.5)   # se obisnuieste cu zgomotul din jur
        print("Ascult...")
        try:
            audio = r.listen(sursa, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return ""   # n-a zis nimic in 5 secunde

    try:
        return r.recognize_google(audio, language=config.LIMBA)
    except sr.UnknownValueError:
        return ""   # n-a inteles ce s-a spus
