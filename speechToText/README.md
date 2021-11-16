### IBM key
Please check the last section of the handover report for IBM key and Instance ID

### Wake word engine
We use the off-the-shelf [Porcupine Wake Word Engine](https://picovoice.ai/platform/porcupine/) for wake word detection. We have trained a custom wake phrase, namely, "Hey Reachy", using [Picovoice Console](https://picovoice.ai/console/).

### Codes

1. <code>stream_recognition.py</code>
```angular2html
To use the IBM speech recognition API, first install websocket-client and pyAudio
$ pip install websocket-client==0.56.0
$ pip install pyaudio

To run a live speech recognition
$ python stream_recognition.py 
optional argument:
$ python stream_recognition.py [-t timeout]
The program will be active for a period, and print out transcripts in real time. The default timeout period is 60 seconds.
```

2. Wake word engine

Installation
```angular2html
$ pip install pvporcupinedemo
```

To detect custom keyword "Hey Reachy" (e.g. models created using [Picovoice Console](https://picovoice.ai/console/))
use `keyword_paths` argument
```angular2html
$ python porcupine_demo_mic.py --keyword_paths ${KEYWORD_PATH_ONE}
```

3. <code>textToSpeech.py</code>
```
To use the text-to-speech service, first install pyttsx3
$ pip install pyttsx3

To run the text-to-speech
$ python textToSpeech.py
```
