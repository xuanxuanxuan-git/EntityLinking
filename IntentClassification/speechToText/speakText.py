# Author name: Yueqing Xuan
# Student ID: 1075355

# The program calls a text-to-speech engine which reads out text input.
# In order to use it, a package must be installed through:
#   pip3 install pyttsx3

import pyttsx3

def initialise_engine():
    engine = pyttsx3.init()
    # rate = engine.getProperty('rate')  # getting details of current speaking rate
    # print(rate)
    engine.setProperty('rate', 150)

    # volume = engine.getProperty('volume')  # getting to know current volume level (min=0 and max=1)
    # print(volume)  # printing current volume level
    engine.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    return engine


def speak(engine, text):
    engine.say(text)
    engine.runAndWait()


# engine = initialise_engine()
# speak(engine, "I will speak this text")
