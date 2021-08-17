# Author name: Yueqing Xuan
# Student ID: 1075355

# This program will listen to the voice command, transcribe the
#   audio data to the text.
# The recognizer will attempt multiple times if it cannot understand
#   the voice command.

# PRE-REQUISITES
# The program calls a text-to-speech engine which reads out text input.
# In order to use it, a package must be installed through:
#   pip3 install pyttsx3

import os
import sys
import configparser
import speech_recognition as sr
rootpath=os.path.abspath('..')   # Represents the absolute path of the parent folder (IntentClassification)
sys.path.append(rootpath)
# from textblob import TextBlob
from speechToText.textToSpeech import initialise_engine, speak

# number of attempts allowed
PROMPT_LIMIT = 3


def get_audio(recognizer, microphone, engine, attempt_limit):
    """
    transcribe speech that are recorded from microphone
    :return:  returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occurred, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """

    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("recognizer must be Recognizer instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("microphone must be Microphone instance")

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    with microphone as source:
        print("A few seconds of silence,please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=3)
        # print("Listening...")
        # speak(engine, "Hi, how can I help you?")

        # greeting = recognizer.listen(source, timeout=5)
        # activation = recognizer.recognize_google(greeting)
        # activation = activation.lower()
        # print(activation)
        # if there_exist(activation):
        #     speak(engine, "Hi, how can I help you")
        #     print("Hi, how can I help you?")

        # recognize speech using IBM Speech to Text
        IBM_USERNAME = "apikey"
        IBM_PASSWORD = get_auth()

        while True:
        # for i in range(attempt_limit):
            print("listening...")
            audio = recognizer.listen(source, timeout=4)
            print("transcribing...")

            try:
                text = recognizer.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD)
                response["transcription"] = text
                break

            # API was unreachable or unresponsive
            except sr.RequestError as e:
                response["success"] = False
                response["error"] = e
                break

            # if the audio can not be understood, retry for a limited number of attempts
            except sr.UnknownValueError as e:
                response["error"] = "Unable to understand audio"
                # if i < attempt_limit - 1:
                #     print("Attempt left: {}.".format(attempt_limit - i - 1))
                #     speak(engine, "Sorry I don't understand")
                # else:
                #     print("exiting...")

    return response


def get_auth():
    config = configparser.RawConfigParser()
    config.read('speech.cfg')
    apikey = config.get('auth', 'apikey')
    return apikey


def recognize_speech_from_mic(attempt_limit=PROMPT_LIMIT):
    # obtain audio from the microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    engine = initialise_engine()  # text to speech engine
    sentence = None

    response = get_audio(recognizer, microphone, engine, attempt_limit)

    # if the audio can be successfully recognised, return the sentence
    if response["transcription"]:
        # print("Google Speech Recognition thinks you said " + response["transcription"])
        sentence = response["transcription"].lower()

    # if no voice recognised after all attempts
    elif response["success"] and not response["transcription"]:
        print("No voice recognised")

    # if the API is unavailable
    elif not response["success"]:
        print("Error: {}".format(response["error"]))

    return sentence


if __name__ == "__main__":
    message = recognize_speech_from_mic()
    print(message)

    # blob = TextBlob(message).correct()
    # print(blob)