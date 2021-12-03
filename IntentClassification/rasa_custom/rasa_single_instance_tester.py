# Author : Karun Mathew
# Student Id: 1007247

# This program posts a single command to the RASA server for intent classification

# PRE-REQUISITES
# This program expects the RASA server to be running at the location specified in apputil.RASA_SERVER
#
# To start a RASA Server, please run the below command
# rasa run --enable-api -m models/20201123-111915.tar.gz
# Please replace the above command with the appropriate trained model

import requests
import json
import os
import sys
import socket
import time
# Represents the absolute path of the parent folder (IntentClassification)
rootpath = '/mnt/c/Users/62572/Desktop/COMP90055/IntentClassifier/IntentClassification'
# reachy_path = '/mnt/c/Users/62572/Desktop/COMP90055/IntentClassifier/reachy'
intent_input_path = '/mnt/c/Users/62572/Desktop/COMP90055/IntentClassifier/IntentClassification/rasa_custom' \
                    '/intent_input.txt'
intent_prediction_path = '/mnt/c/Users/62572/Desktop/COMP90055/IntentClassifier/IntentClassification/rasa_custom' \
                         '/intent_output.txt'
sys.path.append(rootpath)
# sys.path.append(reachy_path)

from util.apputil import RASA_SERVER


headers = {
    'Content-type': 'application/json'
}


# posts a single command instance to the RASA server
# the data includes both language and visual data
# delimited by the LANG_VISUAL_DELIMITER specified in apputil
def post_to_rasa(command):

    data = '{"text": "' + command + '"}'
    response = requests.post(RASA_SERVER, headers=headers, data=data)
    response_json = json.loads(response.text)

    intent = response_json['intent']['name']
    confidence = response_json['intent']['confidence']
    intent_ranking = response_json['intent_ranking']

    print('\nCommand          : ', command)
    print('Predicted Intent : ', intent)
    print('Confidence Score : ', confidence)
    print('Intent Rankings  : ', intent_ranking)

    return intent, confidence, intent_ranking


# write the prediction results to a file
def write_prediction_to_file(command, output_path):
    intent_prediction = ''
    intent_prediction, _, _ = post_to_rasa(command)
    with open(output_path, "w") as f:
        f.write(intent_prediction)
        f.close()
    return intent_prediction


# read the language and visual data from a file
def read_intent_input(input_file_path, output_path):
    command = ''
    while True:
        if os.path.isfile(input_file_path):
            f = open(input_file_path, "r")
            command = f.readline()
            prediction = write_prediction_to_file(command, output_path)
            f.close()
            os.remove(input_file_path)
            break
        else:
            # print("not found")
            time.sleep(0.1)
    return True


# Sample test instances
if __name__ == "__main__":

    # language data
    post_to_rasa('pick up the phone and then put it down')

    # With both language and visual data
    post_to_rasa('pick up the credit card on the table @@@@@@ 0.21 3.76 6.87 0.87')
    post_to_rasa('go up to the table in front of you pick up the credit card to the left of the cardboard box on the table turn to your left and walk into the living room then turn to the first arm chair on your right @@@@@@ 0.79 2.0 2.67 0.978')

    # read the command message from a file
    # read_intent_input(intent_input_path, intent_prediction_path)