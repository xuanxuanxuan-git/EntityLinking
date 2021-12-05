# Yueqing Xuan
# 1075355

# This file performs the entity recognition and entity linking.
# It first loads the trained LSTM model, and then takes a textual command and performs
# the entity recognition. Then it finds all the visual and textual co-reference links
# and then get the visual information that is needed by the intent classifier.
import socket
import spacy
import numpy as np
import torch
import os
import sys
import math
import time
import pandas as pd
from urllib3.exceptions import NewConnectionError
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from receiver import *

SPEECH_RECOGNITION_PATH = 'C:/Users/62572/Desktop/COMP90055/IntentClassifier/speechToText/wake_word'
PARENT_FOLDER_PATH = 'C:/Users/62572/Desktop/COMP90055/IntentClassifier/EntityLinker'
INTENT_CLASSIFICATION = 'C:/Users/62572/Desktop/COMP90055/IntentClassifier/IntentClassification/rasa_custom'
REACHY_PATH = 'C:/Users/62572/Desktop/COMP90055/IntentClassifier/reachy'
INTENT_INPUT_FILE = 'C:/Users/62572/Desktop/COMP90055/IntentClassifier/IntentClassification/rasa_custom/intent_input.txt'
INTENT_OUTPUT_FILE = 'C:/Users/62572/Desktop/COMP90055/IntentClassifier/IntentClassification/rasa_custom/intent_output.txt'

sys.path.append(SPEECH_RECOGNITION_PATH)
sys.path.append(PARENT_FOLDER_PATH)
sys.path.append(INTENT_CLASSIFICATION)
sys.path.append(REACHY_PATH)

from wake_word_engine import *
from linkerUtil import MODEL_PATH, TRAIN_DEP_FILE_PATH, DEV_DEP_FILE_PATH, TEST_DEP_FILE_PATH
from reachy_start import *
from intent_to_reachy import *

# --------------------------- Global parameters ------------------------------- #

# assign each label with a unique index
roleset_size = 5
role_to_idx = {"O": 0, "B-TARGET": 1, "I-TARGET": 2, "B-RECEPTACLE": 3,
               "I-RECEPTACLE": 4}

depset_size = 41
embedding_dim = 300

# parameters used in EntityLinker model
feature_dimension = depset_size + embedding_dim
hidden_dimension = 50

# parameters related to the camera setup
DIRECTION = "opposite"              # "right", "left", "same"
TRANSLATION = [0.77, -0.1, 0.15]    # [dx, dy, dz]
THETA = 30                          # [0, 90]


# preprocess dependency parsing relations using one-hot encoding
# Collect all the dependency relations that occur in all the datasets
def collect_dep_relations(input_files):
    dep_set = set()

    for file in input_files:
        with open(file, "r") as f:
            for line in f:
                dep_list = line.split()
                for dep in dep_list:
                    dep_set.add(dep)

    return dep_set


# Transform the categorical column into their numerical counterparts, via the
# one-hot encoding scheme.
def create_dep_encoding(dep_set):
    ids = []
    id = 0
    dep_list = list(dep_set)

    for dep in dep_list:
        ids.append(id)
        id += 1

    df = pd.DataFrame(list(zip(ids, dep_list)), columns=['Id', 'Dep'])
    dep_df = pd.get_dummies(df.Dep)

    return dep_df


# Return the vector of a dependency relation
# Output is a 41-dimensional vector
def get_dep_vector(dep_df, dep_string):
    try:
        return np.array(dep_df[dep_string].tolist(), dtype=np.float32)
    except KeyError as e:
        return np.array(dep_df['FOOT'].tolist(), dtype=np.float32)
        # print("Error: Dependency relation not found")
        # return None


# collect the dependency relation set, and represent each dep relation using
# one-hot encoding
dep_set = collect_dep_relations({TRAIN_DEP_FILE_PATH, DEV_DEP_FILE_PATH, TEST_DEP_FILE_PATH})
dep_df = create_dep_encoding(dep_set)


# ----------------------------------------------------------------------------- #

# Initialise a new entity linker model that has the same architecture as the trained LSTM model
class EntityLinker(nn.Module):

    def __init__(self, feature_dim, hidden_dim, roleset_size):
        super(EntityLinker, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.roleset_size = roleset_size

        self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True)
        self.cls_layer = nn.Linear(self.hidden_dim, self.roleset_size)

    def forward(self, input_features, input_lens):
        input_packed = pack_padded_sequence(input_features,
                                            input_lens, batch_first=True,
                                            enforce_sorted=False)

        lstm_out_packed, _ = self.lstm(input_packed)

        # lstm_out_padded shape: batch size x batch_max_len x lstm_hidden_dim
        lstm_out_padded, output_lengths = pad_packed_sequence(lstm_out_packed,
                                                              batch_first=True)

        # label_out size: batch_size x batch_max_len x roleset_size
        label_out = self.cls_layer(lstm_out_padded)

        return label_out


# map the index of the label to label in the label set
def idx_to_label(preds, role_to_idx):
    preds_string = []
    for result in preds:
        role = list(role_to_idx.keys())[list(role_to_idx.values()).index(result)]
        preds_string.append(role)
    return preds_string


# process the textual command by:
# tokenize the sentence, then get the word embedding features and dependency relation features
# for each token in the sentence
def process_sentence(nlp, sentence_string):
    sentence_nlp = nlp(sentence_string.lower())
    nlp_trf = spacy.load("en_core_web_trf", disable=["ner"])
    sentence_nlp_trf = nlp_trf(sentence_string.lower())

    num_token = len(sentence_nlp)
    tensor_shape = (num_token, feature_dimension)
    tokens_features_t = torch.zeros(tensor_shape)

    i = 0
    for token in sentence_nlp:
        dep = get_dep_vector(dep_df, sentence_nlp_trf[i].dep_)
        vector = token.vector
        feature = np.append(vector, dep)
        feature_tensor = torch.from_numpy(feature).float()
        tokens_features_t[i] = feature_tensor
        i += 1

    return tokens_features_t


# get the sequence labeling predictions of the textual command
def predict(model, nlp, sentence_string, role_to_idx):
    model.eval()
    features_t = process_sentence(nlp, sentence_string)
    input_lens = features_t.shape[0]

    input = features_t[None, :, :]
    predictions = []

    with torch.no_grad():
        label_scores = model(input, [input_lens])
        role_scores = F.log_softmax(label_scores, dim=2)
        flatten_scores = role_scores.view(-1, label_scores.shape[2])

        for i in range(input_lens):
            pred = torch.argmax(flatten_scores[i])

            predictions.append(pred.item())

        pred_roles = idx_to_label(predictions, role_to_idx)
    return pred_roles


# based on the sequence labeling predictions, combine the entity labels of the
# same entity into one entity.
# for example, combine credit(B-TARGET) and card(I-TARGET) into credit card (target)
def get_target_and_recep(sentence_nlp, pred_roles):
    target = []
    receptacle = []

    i = 0
    for label in pred_roles:
        token = sentence_nlp[i].text
        if label == 'B-TARGET':
            target.append(token)
        elif label == 'I-TARGET':
            item = target[-1] + ' ' + token
            target[-1] = item
        elif label == 'B-RECEPTACLE':
            receptacle.append(token)
        elif label == 'I-RECEPTACLE':
            item = receptacle[-1] + ' ' + token
            receptacle[-1] = item
        i += 1

    return target, receptacle


# perform the entity recognition
# take the textual command and return the target entity and receptacle entity
def command_to_entities(sentence):
    sentence_nlp = nlp(sentence.lower())

    result = predict(model_load, nlp, sentence, role_to_idx)
    target, receptacle = get_target_and_recep(sentence_nlp, result)

    print("\nThe command you entered is: ", sentence)
    print("target:     {} \nreceptacle: {}\n".format(target, receptacle))

    return target, receptacle


# Match the textual entity with the most similar visual mention of entity from the visual pipeline.
# Similarity is measured by calculating the cosine similarity between word vectors.
# The threshold is 0.5, meaning that if the similarity is greater than 0.5, then there is
# a visual and textual co-reference link.
def find_most_similar_visual_item(nlp, entity_list, textual_entity_name):
    textual_entity = nlp(textual_entity_name)
    max_sim = 0.5
    max_coco = ''

    for item in entity_list:
        if item == "diningtable":
            item = "dining table"
        elif item == "pottedplant":
            item = "potted plant"

        visual_entity = nlp(item)

        if visual_entity[0].has_vector:
            sim = textual_entity.similarity(visual_entity)
            if sim > max_sim:
                max_sim = sim
                max_coco = item
        else:
            print("{} not found from the visual pipeline".format(textual_entity_name))

    return max_coco


# ------------------------------ for Intent classification -----------------------#

def get_dot_product(x, y):
    return round(np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y))), 4)


def get_agent_facing_direction_vector(agent_face_direction):
    z = round(math.cos(math.radians(agent_face_direction)), 2)
    x = round(math.sin(math.radians(agent_face_direction)), 2)
    return [x, z]


# gets the angle between the object and the direction the agent is facing
def get_dot_product_score(agent_pos, object_pos, agent_face_direction):
    f = get_agent_facing_direction_vector(agent_face_direction)
    o_a = np.subtract([object_pos[0], object_pos[2]], [agent_pos[0], agent_pos[2]])
    if o_a[0] == 0 and o_a[1] == 0:
        return 1
    else:
        return get_dot_product(f, o_a)


# Calculate the distance between the target entity and the receptacle entity
def calculate_distance(visual_info, target, recep):
    (x1, y1, z1) = (0, 0, 0)
    (x2, y2, z2) = (0, 0, 0)
    distance_1 = -1
    distance_2 = -1
    distance_3 = -1

    if target:
        (x1, y1, z1) = target["position"]
        for entity in visual_info:
            if entity["name"] == target["name"]:
                z1 = entity["position"][2]
                break
        distance_1 = round(math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2), 2)

    if recep:
        (x2, y2, z2) = recep["position"]
        for entity in visual_info:
            if entity["name"] == recep["name"]:
                z2 = entity["position"][2]
                break
        distance_2 = round(math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2), 2)

    if target and recep:
        distance_3 = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2), 2)

    angle = get_dot_product_score([0, 0, 0], [x1, y1, z1], 0)

    return str(distance_1) + ' ' + str(distance_2) + ' ' + str(distance_3) + ' ' + str(angle)


# write the command to a file
# The intent classifier will read the command in the file and perform intent prediction
def write_to_file(file_path, content):
    f = open(file_path, "w")
    f.write(content)
    f.close()


# read the intent prediction result from the file
# The prediction result is produced from the intent classifier (running in a separate environment)
def read_intent_prediction(file_path):
    intent_prediction = ''
    while True:
        if os.path.isfile(file_path):
            f = open(file_path, "r")
            intent_prediction = f.read()
            print("Predicted intent: {}\n".format(intent_prediction))
            f.close()
            os.remove(file_path)
            break
        else:
            # print("not found")
            time.sleep(0.1)
    return intent_prediction


# -------------------------- perform the entire process --------------------------- #
if __name__ == '__main__':
    # load the saved model and the spacy's language model
    model_load = EntityLinker(feature_dim=feature_dimension,
                              hidden_dim=hidden_dimension,
                              roleset_size=roleset_size)
    model_load.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    mode = input("Enter 1 to type a command, enter 2 to speak. ")

    while True:
        try:
            # ------------- entity recognition ------------------------ #
            if mode == '1':
                sentence = input("Enter a command: ")

            elif mode == '2':
                audio_input = microphone_stt()
                sentence = audio_input
                if not audio_input:
                    raise Exception("No audio input")
                print("Audio input: ", audio_input)

            else:
                print("raise error")
                raise KeyError
            target, receptacle = command_to_entities(sentence)

            try:
                # ------------- match with visual pipeline ------------------------ #

                # get visual info from the visual pipeline
                visual_info = receive_from_zed()

                if visual_info:
                    transformed_visual_info = coordinate_transformation(visual_info, DIRECTION, THETA, TRANSLATION)
                    visual_entity_list = [item["name"] for item in visual_info]
                    print("Entities detected by the visual pipeline: ", visual_entity_list)

                    target_coord = None
                    recep_coord = None
                    visual_target_entity = None
                    visual_recep_entity = None

                    # if there is a target in the command
                    if target:
                        visual_target_name = find_most_similar_visual_item(nlp, visual_entity_list, target[0])
                        for item in transformed_visual_info:
                            if item["name"].replace(' ', '') == visual_target_name.replace(' ', ''):
                                visual_target_entity = item
                                target_coord = item["position"]
                                break
                    if receptacle:
                        visual_recep_name = find_most_similar_visual_item(nlp, visual_entity_list, receptacle[0])
                        for item in transformed_visual_info:
                            if item["name"].replace(' ', '') == visual_recep_name.replace(' ', ''):
                                visual_recep_entity = item
                                recep_coord = item["position"]
                                break

                else:   # if no objects detected
                    raise ValueError

                # ------------- pass the visual info to intent classifier ---------- #
                message = sentence
                if visual_info:
                    message = sentence + ' @@@@@@ ' + calculate_distance(visual_info, visual_target_entity,
                                                                         visual_recep_entity)

                print("Input to the intent classifier: {}\n".format(message))
                write_to_file(INTENT_INPUT_FILE, message)
                intent = read_intent_prediction(INTENT_OUTPUT_FILE)

                #  ----------- connect to reachy and perform the task ------------- #
                try:
                    reachy_robot = connect_to_robot(ROBOT_IP)
                    execute_reachy(reachy_robot, intent, target_coord, recep_coord)

                except Exception:
                    raise ConnectionError

            except RuntimeError:
                print("visual pipeline is off")
            except (ValueError, AttributeError):
                print("Objects not detected by the visual pipeline")
            except ConnectionError:
                print("Failed to connect to Reachy")

        except KeyError:
            print("Mode not supported")
        except KeyboardInterrupt:
            print("\nQuit!")
            if os.path.isfile(INTENT_INPUT_FILE):
                os.remove(INTENT_INPUT_FILE)
        finally:
            sys.exit(1)

