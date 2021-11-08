import os
import json
import linecache
import numpy as np
import spacy
from spacy.symbols import *
import sys

PARENT_FOLDER_PATH = 'C:/Users/62572/Desktop/COMP90055/IntentClassifier/EntityLinker'
sys.path.append(PARENT_FOLDER_PATH)
from linkerUtil import ANNOTATED_TRAIN_DATA_PATH, ANNOTATED_DEV_DATA_PATH, ANNOTATED_TEST_DATA_PATH, EXTRA_TRAIN_DATA


# replace the existing target/receptacle entities with the new entity
def replace_entity(input_file, start_index, new_entity, entity_type):
    index = start_index
    output_file = open(EXTRA_TRAIN_DATA, "a")

    for i in range(index, index + 400):
        line = linecache.getline(input_file, i).rstrip()
        json_object = json.loads(line)
        if json_object['entities']:
            for entity in json_object['entities']:

                if entity['role'] == entity_type:
                    if entity_type == 'target' and len(entity['value'].split()) != len(new_entity.split()):
                        continue

                    old_entity = entity['value']
                    old_length = entity['end_token'] - entity['start_token']
                    old_char_length = entity['end_char'] - entity['start_char']

                    entity['value'] = new_entity
                    entity['label'] = new_entity
                    entity['end_token'] = entity['start_token'] + len(new_entity.split())
                    entity['end_char'] = entity['start_char'] + len(new_entity)

                    json_object['text'] = json_object['text'].replace(old_entity, new_entity)

                    if entity_type == 'target':
                        if len(json_object['entities']) == 2:
                            recep_entity = json_object['entities'][1]
                            recep_entity['start_token'] = recep_entity['start_token'] - (old_length - len(new_entity.split()))
                            recep_entity['end_token'] = recep_entity['end_token'] - (old_length - len(new_entity.split()))
                            recep_entity['start_char'] = recep_entity['start_char'] - (old_char_length - len(new_entity))
                            recep_entity['end_char'] = recep_entity['end_char'] - (old_char_length - len(new_entity))

                            json_object['entities'][1] = recep_entity
                    output_file.write(json.dumps(json_object))
                    output_file.write('\n')

    output_file.close()


ALL_SENTENCES = "Generated_dataset/shuffled_dataset/all_sentences.txt"
ALL_DEV = "Generated_dataset/shuffled_dataset/all_dep.txt"
ALL_LABELS = "Generated_dataset/shuffled_dataset/all_labels.txt"


# shuffle the dataset
def shuffle_dataset():
    f = open(ALL_SENTENCES, "r")
    total_num = len(f.readlines())
    np.random.seed(1)
    all_indices = np.arange(total_num)
    np.random.shuffle(all_indices)


# replace_entity(ANNOTATED_DEV_DATA_PATH, 1, "laptop", "receptacle")
# replace_entity(ANNOTATED_DEV_DATA_PATH, 401, "suitcase", "receptacle")
# replace_entity(ANNOTATED_DEV_DATA_PATH, 801, "handbag", "receptacle")
# replace_entity(ANNOTATED_DEV_DATA_PATH, 1201, "book", "receptacle")
#
#
# replace_entity(ANNOTATED_DEV_DATA_PATH, 1601, "keyboard", "target")
# replace_entity(ANNOTATED_DEV_DATA_PATH, 2101, "scissor", "target")
# replace_entity(ANNOTATED_DEV_DATA_PATH, 2601, "umbrella", "target")
# replace_entity(ANNOTATED_DEV_DATA_PATH, 3141, "mouse", "target")

print("completed")
