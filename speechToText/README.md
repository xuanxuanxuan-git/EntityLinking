### Wake word engine and speech recognition
We use the off-the-shelf [Porcupine Wake Word Engine](https://picovoice.ai/platform/porcupine/) 
for wake word detection. We have trained a custom wake phrase, namely, "Hey Reachy", 
using [Picovoice Console](https://picovoice.ai/console/). You can look at their 
[github repo](https://github.com/watson-developer-cloud/python-sdk) for sample codes.


In addition, we use the off-the-shelf [IBM Watson speech recognition service](https://cloud.ibm.com/docs/speech-to-text). 
You can also find some sample codes provided in their [github repo](https://github.com/watson-developer-cloud/python-sdk).
<br><hr>
### Codes

1. <code>wake_work/wake_word_engine.py</code>

This file detects the wake word (phrase) "Hey Reachy". Then it will notify the user by a short beep sound 
after the wake ward is detected. Then the IBM watson speech recognition service will be activated 
and start to transcribe the verbal command. 

<b>Wake word engine installation</b>
```angular2html
$ pip install pvporcupinedemo
```

<b>IBM Watson speech recognition installation</b>
```angular2html
$ pip install websocket-client==0.56.0

You don't need to install pyaudio in this program since we use pv_recorder (included in the 
pvporcupinedemo package) as the audio source. If you would like to run the speech recognition 
tool alone, please follow the instructions for stream_recognition.py.
```

<b>Short beep sound notification</b>
```angular2html
Winsound module is specially made for Windows. You may need to use other module to generate 
this specific sound if you are using different operating system. 

The codes related to this function are in line 158 to line 160. 
```

<b>Wake word model</b>
```angular2html
To use the custom wake word model, we first need to train it using the picovoice console.
Please follow the instruction in the OneNote (RA week 3 -> wake word engine) on how to complete
this step. 

After obtaining the trained model, you need to alter the code in line 58 with the new path to
the model. The generated model is only valid for 30 days, but you can re-train it for free 
(for personal use only).

Please also note that the wake word model provided in this repo can only be used on Windows system.
If you are using a different operating system, you need to train a new model as well. 
```

#### IBM Watson speech recognition credentials


You'll need to sign up for the Watson STT service and use the API key and Instance ID to 
connect to the Watson streaming server. You can find the credentials in the second last section 
of the handover report as well as in the OneNote. 

To use the credential, first create a file named <i>speech.cfg</i>. Copy the informtion from 
<i>speech.cfg.example</i> to this file. Fill in the <i>apikey</i> and the <i>instance_id</i> accordingly.

<b>Run the wake word engine</b>
```angular2html
This program is invoked directly from the main function in the entity_recognition.py.

To test this program, you can also run by:
$ python wake_word_engine.py

optional arguments are:
$ python wake_word_engine.py [--output_path] [--audio_device_index] [--show_audio_devices]
```

<br><hr>
2. <code>stream_recognition.py</code>

This files contains codes that uses IBM Watson speech recognition service only. You will also 
need to provide credentials.

```angular2html
To use the IBM speech recognition API, first install websocket-client and pyAudio
$ pip install websocket-client==0.56.0

a) Installing pyAudio on Windows
$ pip install pipwin
$ pipwin install pyaudio

b) Installing pyAudio on Mac OS with homebrew
$ brew install portaudio
$ pip install pyaudio

c) Installing pyAudio on linux
$ pip install pyaudio
```

To run a live speech recognition
```angular2html
$ python stream_recognition.py 
optional argument:
$ python stream_recognition.py [-t timeout]
```
The program will be active for a period, 
and print out transcripts in real time. The default timeout period is 60 seconds.

