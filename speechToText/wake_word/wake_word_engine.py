#
# Copyright 2018-2021 Picovoice Inc.
#
# You may not use this file except in compliance with the license. A copy of the license is located in the "LICENSE"
# file accompanying this source.
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

# This file contains the codes of wake word engine (Porcupine) and speech recognition (IBM watson).
#

import argparse
import os
from datetime import datetime
# from threading import Thread
import numpy as np
import pvporcupine
import soundfile
from pvrecorder import PvRecorder

# modules for IBM Watson stream speech recognition
import base64
import configparser
import json
import threading
import time
import pyaudio
import websocket
from websocket._abnf import ABNF
import sys
import struct
import winsound

CHUNK = 512
FORMAT = pyaudio.paInt16
# Even if your default input is multi channel (like a webcam mic),
# it's really important to only record 1 channel, as the STT service
# does not do anything useful with stereo. You get a lot of "hmmm"
# back.
CHANNELS = 1
# Rate is important, nothing works without it. This is a pretty
# standard default. If you have an audio device that requires
# something different, change this.
RATE = 16000
RECORD_SECONDS = 5
FINALS = []
LAST = None

# Parameters for notification sound (after the wake word is detected)
DURATION = 600  # in milliseconds
FREQ = 400      # in Hz
rootpath = '../../'
config_file = os.path.join(rootpath, 'speechToText/speech.cfg')
# config_file = '../../speech.cfg'
KEYWORD_PATH = os.path.join(rootpath, 'speechToText/wake_word/hey_reachy_windows/hey-reachy__en_windows_2021-12-16-utc_v1_9_0.ppn')


class PorcupineDemo(threading.Thread):
    """
    Microphone Demo for Porcupine wake word engine. It creates an input audio stream from a microphone, monitors it, and
    upon detecting the specified wake word(s) prints the detection time and wake word on console. It optionally saves
    the recorded audio into a file for further debugging.
    """

    def __init__(
            self,
            library_path,
            model_path,
            keyword_paths,
            sensitivities,
            ibm_connection,     # websocket connection
            input_device_index=None,
            output_path=None):

        """
        Constructor.

        :param library_path: Absolute path to Porcupine's dynamic library.
        :param model_path: Absolute path to the file containing model parameters.
        :param keyword_paths: Absolute paths to keyword model files.
        :param sensitivities: Sensitivities for detecting keywords. Each value should be a number within [0, 1]. A
        higher sensitivity results in fewer misses at the cost of increasing the false alarm rate. If not set 0.5 will
        be used.
        :param input_device_index: Optional argument. If provided, audio is recorded from this input device. Otherwise,
        the default audio input device is used.
        :param output_path: If provided recorded audio will be stored in this location at the end of the run.
        """

        super(PorcupineDemo, self).__init__()

        self._library_path = library_path
        self._model_path = model_path
        self._keyword_paths = keyword_paths
        self._sensitivities = sensitivities
        self._input_device_index = input_device_index

        self._output_path = output_path
        if self._output_path is not None:
            self._recorded_frames = []

        self.ws = ibm_connection

    def run(self):
        """
         Creates an input audio stream, instantiates an instance of Porcupine object, and monitors the audio stream for
         occurrences of the wake word(s). It prints the time of detection for each occurrence and the wake word.
         """

        keywords = list()
        for x in self._keyword_paths:
            keyword_phrase_part = os.path.basename(x).replace('.ppn', '').split('_')
            if len(keyword_phrase_part) > 6:
                keywords.append(' '.join(keyword_phrase_part[0:-6]))
            else:
                keywords.append(keyword_phrase_part[0])

        porcupine = None
        recorder = None
        connected = True
        try:
            porcupine = pvporcupine.create(
                library_path=self._library_path,
                model_path=self._model_path,
                keyword_paths=self._keyword_paths,
                sensitivities=self._sensitivities)

            recorder = PvRecorder(device_index=self._input_device_index, frame_length=porcupine.frame_length)
            recorder.start()
            activated = False
            result = -1

            print(f'Using device: {recorder.selected_device}')

            print('Listening {')
            for keyword, sensitivity in zip(keywords, self._sensitivities):
                print('  %s (%.2f)' % (keyword, sensitivity))
            print('}')

            while True:
                pcm = recorder.read()   # output is a list containing audio frames (length = 512)

                if activated:
                    # self._recorded_frames.append(pcm)
                    try:
                        data = struct.pack('h'*porcupine.frame_length, *pcm)
                        self.ws.send(data, ABNF.OPCODE_BINARY)
                    except websocket._exceptions.WebSocketConnectionClosedException:
                        connected = False
                        break

                if not activated:
                    result = porcupine.process(pcm)

                if result >= 0:
                    duration = DURATION  # milliseconds
                    freq = FREQ  # Hz
                    winsound.Beep(freq, duration)

                    data = struct.pack('h' * 512, *pcm)
                    self.ws.send(data, ABNF.OPCODE_BINARY)
                    print('[%s] Detected %s' % (str(datetime.now()), keywords[result]))

                    activated = True
                    result = -1


        except KeyboardInterrupt:
            print('Stopping ...')
        finally:
            if porcupine is not None:
                porcupine.delete()

            if recorder is not None:
                recorder.delete()

            if connected:
                data = {"action": "stop"}
                self.ws.send(json.dumps(data).encode('utf8'))
                # ... which we need to wait for before we shutdown the websocket
                time.sleep(1)
                self.ws.close()

            if self._output_path is not None and len(self._recorded_frames) > 0:
                recorded_audio = np.concatenate(self._recorded_frames, axis=0).astype(np.int16)
                soundfile.write(self._output_path, recorded_audio, samplerate=porcupine.sample_rate, subtype='PCM_16')

    @classmethod
    def show_audio_devices(cls):
        devices = PvRecorder.get_audio_devices()

        for i in range(len(devices)):
            print(f'index: {i}, device name: {devices[i]}')


def wake_word_engine(ws):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--keywords',
        nargs='+',
        help='List of default keywords for detection. Available keywords: %s' % ', '.join(sorted(pvporcupine.KEYWORDS)),
        choices=sorted(pvporcupine.KEYWORDS),
        metavar='')

    parser.add_argument(
        '--sensitivities',
        nargs='+',
        help="Sensitivities for detecting keywords. Each value should be a number within [0, 1]. A higher " +
             "sensitivity results in fewer misses at the cost of increasing the false alarm rate. If not set 0.5 " +
             "will be used.",
        type=float,
        default=None)

    parser.add_argument('--audio_device_index', help='Index of input audio device.', type=int, default=-1)

    parser.add_argument('--output_path', help='Absolute path to recorded audio for debugging.', default=None)

    parser.add_argument('--show_audio_devices', action='store_true')

    args = parser.parse_args()

    if args.show_audio_devices:
        PorcupineDemo.show_audio_devices()
    else:
        keyword_paths = [KEYWORD_PATH]

        if args.sensitivities is None:
            args.sensitivities = [0.5] * len(keyword_paths)

        if len(keyword_paths) != len(args.sensitivities):
            raise ValueError('Number of keywords does not match the number of sensitivities.')

        PorcupineDemo(
            library_path=pvporcupine.LIBRARY_PATH,
            model_path=pvporcupine.MODEL_PATH,
            keyword_paths=keyword_paths,
            sensitivities=args.sensitivities,
            output_path=args.output_path,
            input_device_index=args.audio_device_index,
            ibm_connection=ws).run()


# ----------------- IBM Watson ---------------------------- #


def read_audio(ws, timeout):
    """Read audio and sent it to the websocket port.

    This uses pyaudio to read from a device in chunks and send these
    over the websocket wire.

    """
    # print("* Please speak")
    wake_word_engine(ws)


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
        raise Exception("Speech recognition timed out")


def on_error(ws, error):
    """Print any errors."""
    print("got an error: ", error)
    sys.exit(1)


def on_close(ws):
    """Upon close, print the complete and final transcript."""
    global LAST
    if LAST:
        FINALS.append(LAST)
    transcribe = "".join([x['results'][0]['alternatives'][0]['transcript']
                          for x in FINALS])

    return transcribe


def on_open(ws):
    """Triggered as soon as we have an active connection."""
    # args = ws.args
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
                     args=(ws, int(30))).start()


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


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Transcribe text in real time')
#     parser.add_argument('-t', '--timeout', type=int, default=30)
#     args = parser.parse_args()
#     return args


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
    # ws.args = parse_args()
    # This gives control over the WebSocketApp. This is a blocking
    # call, so it won't return until the ws.close() gets called (after
    # 6 seconds in the dedicated thread).
    ws.run_forever()

    speech = remove_hesitation(on_close(ws))
    return speech


# ------------------------- main process ----------------------- #

if __name__ == '__main__':
    # wake_word_engine(None)
    text = microphone_stt()
    if text:
        print("Text: {}".format(text))
    else:
        print("No audio")
