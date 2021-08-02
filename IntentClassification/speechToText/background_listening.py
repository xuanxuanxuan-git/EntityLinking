# Author name: Yueqing Xuan
# Student ID: 1075355

# ---------- INCOMPLETE ---------------------------

# This program enables continuous background listening
# It can be used for voice activation
# The issue with this function is that background listening cannot
#   stop on keywords (if an activation word is heard).


import speech_recognition as sr
import time


def callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio)
        print(text)
        if "hi" in text:
            print("keyword detected")
            command = recognizer.listen(source, timeout=5)
            print("output from main: " + recognizer.recognize_google(command))

    except sr.UnknownValueError:
        pass
        print("could not understand")
    except sr.RequestError as e:
        print("Error: {}.".format(e))

def main_recognise():
    pass


r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

# start listening in the background (note that we don't have to do this inside a `with` statement)
stop_listening = r.listen_in_background(m, callback)
# `stop_listening` is now a function that, when called, stops background listening

# do some unrelated computations for 5 seconds
for _ in range(50): time.sleep(0.1)  # we're still listening even though the main thread is doing other things
print("stopping")
# calling this function requests that the background listener stop listening
# stop_listening(wait_for_stop=False)

while True: time.sleep(0.1)
