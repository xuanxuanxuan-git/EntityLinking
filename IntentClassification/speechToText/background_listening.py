# Author name: Yueqing Xuan
# Student ID: 1075355

# This program enables continuous background listening
# It can be used for voice activation (wake word detection)
# The issue with this function is that background listening cannot
#   stop on keywords (if an activation word is heard). It will always
#   listen unless interrupted by keyboard.
# Another issue is accuracy on detecting wake words.

import speech_recognition as sr
import time
import os
import sys
rootpath=os.path.abspath('..')   # Represents the absolute path of the parent folder (IntentClassification)
sys.path.append(rootpath)
from rasa_custom.rasa_single_instance_tester import post_to_rasa


def there_exist(text_string):
    triggers = ['hi', 'hey', 'hello', 'richie']     # unable to transcribe 'reachy'
    if any(word in text_string for word in triggers):
        return True
    else:
        return False


def callback(recognizer, audio):
    try:
        print("listening...")
        text = recognizer.recognize_google(audio)
        print(text)
        if there_exist(text.lower()):
            print("wake word detected")
            print("How can I help you?")
            while True:
                command = get_speech()
                if command:
                    # print("command received: " + command)
                    post_to_rasa(command)
                    break

    except sr.UnknownValueError:
        pass
        # print("could not understand")
    except sr.RequestError as e:
        print("Error: {}.".format(e))


def get_speech():
    command = None
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            audio = r.listen(source, timeout=5)

        try:
            command = r.recognize_google(audio)
            # print("Google Speech Recognition thinks you said " + command)
        except sr.UnknownValueError:
            print("I don't understand")
        except sr.RequestError as e:
            print("API unavailable; {0}".format(e))
    except sr.WaitTimeoutError:
        print("I don't understand")
    return command


def always_listening():

    r = sr.Recognizer()
    m = sr.Microphone()

    with m as source:
        print("A few seconds of silence,please wait...")
        r.adjust_for_ambient_noise(source)

    stop_listening = r.listen_in_background(m, callback)


# calling this function requests that the background listener stop listening
# stop_listening(wait_for_stop=False)
always_listening()
while True: pass
