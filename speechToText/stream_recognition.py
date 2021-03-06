#!/usr/bin/env python
#
# Copyright 2016 IBM
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


# This file contains the IBM watson live speech recognition functions.
# The connection to the online service will timeout after 30s or after 2 seconds of
# silence at the end of a speech.

import argparse
import base64
import configparser
import json
import threading
import time
import os
import pyaudio
import sys
import websocket
from websocket._abnf import ABNF

CHUNK = 1024
FORMAT = pyaudio.paInt16
# Even if your default input is multi channel (like a webcam mic),
# it's really important to only record 1 channel, as the STT service
# does not do anything useful with stereo. You get a lot of "hmmm"
# back.
CHANNELS = 1
# Rate is important, nothing works without it. This is a pretty
# standard default. If you have an audio device that requires
# something different, change this.
RATE = 44100
RECORD_SECONDS = 5
FINALS = []
LAST = None
# LAST_ACTIVE_TIME = 0
# SILENCE = 3
rootpath = '../../'
# config_file = os.path.join(rootpath, 'speechToText/speech.cfg')
config_file = 'speech.cfg'


def read_audio(ws, timeout):
    """Read audio and sent it to the websocket port.

    This uses pyaudio to read from a device in chunks and send these
    over the websocket wire.

    """
    global RATE
    p = pyaudio.PyAudio()
    connected = True
    # NOTE(sdague): if you don't seem to be getting anything off of
    # this you might need to specify:
    #
    #    input_device_index=N,
    #
    # Where N is an int. You'll need to do a dump of your input
    # devices to figure out which one you want.
    RATE = int(p.get_default_input_device_info()['defaultSampleRate'])
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Please speak")
    rec = timeout or RECORD_SECONDS

    for i in range(0, int(RATE / CHUNK * rec)):
        data = stream.read(CHUNK)
        # print("Sending packet... %d" % i)
        # NOTE: we're sending raw binary in the stream, we
        # need to indicate that otherwise the stream service
        # interprets this as text control messages.
        try:
            ws.send(data, ABNF.OPCODE_BINARY)
        except websocket._exceptions.WebSocketConnectionClosedException:
            connected = False
            break

        # if (ft - LAST_ACTIVE_TIME) > SILENCE:
        #     # print("stop")
        #     break

    # Disconnect the audio stream
    stream.stop_stream()
    stream.close()
    print("* done recording")

    if connected:
        # In order to get a final response from STT we send a stop, this
        # will force a final=True return message.
        data = {"action": "stop"}
        ws.send(json.dumps(data).encode('utf8'))
        # ... which we need to wait for before we shutdown the websocket
        time.sleep(1)
        ws.close()

    # ... and kill the audio device
    p.terminate()


def on_message(ws, msg):
    """Print whatever messages come in.

    While we are processing any non trivial stream of speech Watson
    will start chunking results into bits of transcripts that it
    considers "final", and start on a new stretch. It's not always
    clear why it does this. However, it means that as we are
    processing text, any time we see a final chunk, we need to save it
    off for later.
    """
    global LAST
    data = json.loads(msg)

    if "results" in data:
        if data["results"][0]["final"]:
            FINALS.append(data)
            LAST = None
        else:
            LAST = data
        # This prints out the current fragment that we are working on
        print(data['results'][0]['alternatives'][0]['transcript'])

    if "error" in data.keys():
        print(data["error"])


def on_error(ws, error):
    """Print any errors."""
    print("got an error: ", error)


def on_close(ws):
    """Upon close, print the complete and final transcript."""
    global LAST
    if LAST:
        FINALS.append(LAST)
    transcribe = "".join([x['results'][0]['alternatives'][0]['transcript']
                          for x in FINALS])

    # print(transcribe)
    return transcribe


def on_open(ws):
    """Triggered as soon as we have an active connection."""
    args = ws.args
    data = {
        "action": "start",
        # this means we get to send it straight raw sampling
        "content-type": "audio/l16;rate=%d" % RATE,
        "continuous": True,
        "interim_results": True,
        "inactivity_timeout": 2,  # in order to use this effectively
        # you need other tests to handle what happens if the socket is
        # closed by the server.
        "word_confidence": True,
        "timestamps": True,
        "max_alternatives": 3,
        "background_audio_suppression": 0.2
    }

    # Send the initial control message which sets expectations for the
    # binary stream that follows:
    ws.send(json.dumps(data).encode('utf8'))
    # Spin off a dedicated thread where we are going to read and
    # stream out audio.
    threading.Thread(target=read_audio,
                     args=(ws, args.timeout)).start()


def get_url():
    config = configparser.RawConfigParser()
    config.read(config_file)

    host = "us-south"
    instance_id = config.get('auth', 'instance_id')
    return ("wss://api.{}.speech-to-text.watson.cloud.ibm.com/"
            "instances/{}/v1/recognize".format(host, instance_id))


def get_auth():
    config = configparser.RawConfigParser()
    config.read(config_file)
    apikey = config.get('auth', 'apikey')
    return ("apikey", apikey)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Transcribe text in real time')
    parser.add_argument('-t', '--timeout', type=int, default=30)
    # parser.add_argument('-d', '--device')
    # parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    return args


def remove_hesitation(text_string):
    text_string = text_string.replace('%HESITATION ', '')
    return text_string


def microphone_stt():
    # Connect to websocket interfaces
    headers = {}
    userpass = ":".join(get_auth())
    headers["Authorization"] = "Basic " + base64.b64encode(
        userpass.encode()).decode()
    url = get_url()

    # If you really want to see everything going across the wire,
    # uncomment this. However realize the trace is going to also do
    # things like dump the binary sound packets in text in the
    # console.
    #
    # websocket.enableTrace(True)
    global FINALS
    if FINALS:
        FINALS = []
    ws = websocket.WebSocketApp(url,
                                header=headers,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.args = parse_args()
    # This gives control over the WebSocketApp. This is a blocking
    # call, so it won't return until the ws.close() gets called (after
    # 6 seconds in the dedicated thread).
    ws.run_forever()

    speech = remove_hesitation(on_close(ws))
    return speech


if __name__ == "__main__":
    text = microphone_stt()
    if text:
        print("Text: {}".format(text))
    else:
        print("No audio")
