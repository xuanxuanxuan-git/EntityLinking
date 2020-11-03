# This program sends the intents to the rasa server that runs
# by default on the 5005 port. The response will contain the
# probability distribution of all the intents

# The rasa server is started by providing the model path
# rasa run --enable-api -m models/20200722-133414.tar.gz

import random
import math
import requests
import json
import copy
from math import sqrt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from util.apputil import get_dot_product_score
from util.apputil import get_visual_information, READ, RASA_SERVER, DEV_DATA_PATH,\
    MAX_ANGLE, WITH_VISUAL, CONSIDER_ROTATION, LABELS, LANG_VISUAL_DELIMITER


headers = {
    'Content-type': 'application/json'
}


# accepts the test / validation dataset file path and tries to predict the intent
# and prints out the statistics
def read_test_data(file_path):
    file = open(file_path, READ)

    count = 0.0
    correct = 0.0
    extra_count = 0.0
    extra_correct = 0.0

    predicted_tags = []
    actual_tags = []
    extra_predicted_tags = []
    extra_actual_tags = []

    for line in file:
        count += 1

        json_object = json.loads(line)
        action_sequence = json_object['action_sequence']
        desc = json_object['desc']
        desc = [item.strip() for item in desc]
        action_sequence_string = ' '.join(action_sequence)
        desc_string = ' '.join(desc)
        desc_string = remove_special_characters(desc_string)

        # if the record_type is not task_desc add the record without visual information
        if json_object['record_type'] != 'task_desc' and WITH_VISUAL:
            text_string = copy.deepcopy(desc_string)
            extra_correct = post_to_rasa(action_sequence_string, extra_actual_tags,
                                         extra_correct, text_string, extra_predicted_tags)
            extra_count += 1

        # test with both visual and language data
        if WITH_VISUAL:
            visual_data = get_visual_information(json_object['scene_description'])

            # if the first action is PickupObject, consider the
            # orientation info as well when deciding the intent
            if action_sequence_string.strip().startswith('PickupObject') or \
                    action_sequence_string.strip().startswith('PutObject'):
                if random.random() < 0.5:
                    visual_data[3] = round(random.uniform(-1, 0), 2)
                else:
                    visual_data[3] = round(random.uniform(0, 1), 2)
                if math.degrees(math.acos(visual_data[3])) > MAX_ANGLE:
                    action_sequence_string = 'RotateAgent ' + action_sequence_string.strip()
                    action_sequence_string = action_sequence_string.strip()

            desc_string = desc_string + ' ' + LANG_VISUAL_DELIMITER
            for visual_info in visual_data:
                desc_string = desc_string + ' ' + str(visual_info)

        correct = post_to_rasa(action_sequence_string, actual_tags, correct, desc_string, predicted_tags)

    print_statistics(actual_tags, correct, count, predicted_tags)
    # Test the performance without visual info for entries that do not need
    # visual info, to check that language is still given importance
    print_statistics(actual_tags + extra_actual_tags, correct + extra_correct,
                     extra_count + count, predicted_tags + extra_predicted_tags)


def print_statistics(actual_tags, correct, count, predicted_tags):
    print('Accuracy :: ', correct / count)
    print('Correct  :: ', correct)
    print('Total    :: ', count)
    print(precision_recall_fscore_support(actual_tags, predicted_tags,
                                          average='macro'))
    print(precision_recall_fscore_support(actual_tags, predicted_tags,
                                          average=None,
                                          labels=LABELS))
    confusion = confusion_matrix(actual_tags, predicted_tags)
    print('Confusion Matrix\n')
    print(confusion)


def remove_special_characters(desc_string):
    desc_string = desc_string.replace('\"', '')
    desc_string = desc_string.replace('.', '')
    desc_string = desc_string.replace(';', '')
    desc_string = desc_string.replace(',', '')
    desc_string = desc_string.lower()
    return desc_string


def post_to_rasa(action_sequence_string, actual_tags, correct, desc_string, predicted_tags):

    data = '{"text": "' + desc_string + '"}'
    response = requests.post(RASA_SERVER, headers=headers, data=data)
    response_json = json.loads(response.text)

    intent = response_json['intent']['name']
    confidence = response_json['intent']['confidence']
    predicted_tags.append(intent.strip())
    actual_tags.append(action_sequence_string.strip())
    if intent.strip() == action_sequence_string.strip():
        correct += 1
    # elif tag == 'NOT_SUPPORTED':
    #    correct += 1
    else:
        print('NOT MATCH : ', data)
        print(response_json)
        print('Predicted Intent :', intent, ' Confidence :', confidence)
        print('Actual Intent :', action_sequence_string)
        print('\n')
    return correct


read_test_data(DEV_DATA_PATH)
