# Author name: Yueqing Xuan
# Student ID: 1075355

# The program calls a text-to-speech engine which reads out text input.
# In order to use it, a package must be installed through:
#   pip3 install pyttsx3

# Problem with pyttsx3: the voice it uses is system dependant
# Alternative solution: to use google Text-to-Speech (gTTS)
# However, gTTS can only work online, and it lead to some latency since it
# needs to generate a mp3 file for the text and play the file using mp3 player

import pyttsx3
import os


def initialise_engine():
    engine = pyttsx3.init()
    # rate = engine.getProperty('rate')  # getting details of current speaking rate
    # print(rate)
    engine.setProperty('rate', 150)

    # volume = engine.getProperty('volume')  # getting to know current volume level (min=0 and max=1)
    # print(volume)  # printing current volume level
    engine.setProperty('volume', 0.9)  # setting up volume level  between 0 and 1

    if os.name == 'nt':
        # for Windows
        # The male voice is "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0"
        voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
        engine.setProperty('voice', voice_id)

    else:   # for Linux and Mac it prints 'posix'
        # for Mac OS
        # change_voice_on_mac(engine, 'en_US', 'VoiceGenderMale')
        change_voice_on_mac(engine, 'en_US', 'VoiceGenderFemale')

    return engine


# change the voice on mac OS
def change_voice_on_mac(engine, language, gender):
    for voice in engine.getProperty('voices'):
        if language in voice.languages and gender == voice.gender:
            engine.setProperty('voice', voice.id)
            return True
    raise RuntimeError("Language '{}' for gender '{}' not found".format(language, gender))


def speak(engine, text):
    engine.say(text)
    engine.runAndWait()


# To test the voice, uncomment all the following lines
engine = initialise_engine()
speak(engine, "Hello, I will speak this text")
