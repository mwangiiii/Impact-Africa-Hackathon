import speech_recognition as sr
recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Adjusting for background noise...Please wait.")
    recognizer.adjust_for_ambient_noise(source)
    print("Listening...Speaking now!")

    try:
        audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Sorry, could not understand audio")
    except sr.RequestError as e:
        print("API request failed")
