### IBM key
Please check the OneNote for IBM key (Github repo -> IBM key)

To use the IBM key, please follow these steps
1. copy the IBM key from OneNote
2. open the <code>speech.cfg.example</code> file, insert the apikey
3. Rename the <code>speech.cfg.example</code> file by removing <i>.example</i>

### Codes
1. <code>recognition.py</code>
```
To run the speech recognition in recognition.py
$ python recognition.py
```
2. <code>stream_recognition</code>
```angular2html
To use the IBM speech recognition API, first install websocket-client
$ pip install websocket-client==0.56.0

To run a live speech recognition
$ python stream_recognition.py 
optional argument:
$ python stream_recognition.py [-t timeout]
The program will be active for a period, and print out transcripts in real time. The default timeout period is 15 seconds.
```

3. <code>textToSpeech.py</code>
```
To use the text-to-speech service, first install pyttsx3
$ pip install pyttsx3

To run the text-to-speech
$ python textToSpeech.py
```
