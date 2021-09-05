import os
import json
import spacy
from spacy.symbols import *
import re

FOLDER_PATH = "C:/Users/62572/Desktop/COMP90055/IntentClassifier/EntityLinker/Data/ALFRED/json_feat_2.1.0/"
TRAIN_DATA_PATH = './Generated_dataset/training_set.txt'
DEV_DATA_PATH = './Generated_dataset/dev_set.txt'
TEST_DATA_PATH = './Generated_dataset/testing_set.txt'

NEW_TRAIN_DATA_PATH = 'Generated_dataset/training_set_new.txt'
NEW_DEV_DATA_PATH = 'Generated_dataset/dev_set_new.txt'
NEW_TEST_DATA_PATH = 'Generated_dataset/testing_set_new.txt'

ANNOTATED_TRAIN_DATA_PATH = 'Generated_dataset/training_set_annotated.txt'
ANNOTATED_DEV_DATA_PATH = 'Generated_dataset/dev_set_annotated.txt'
ANNOTATED_TEST_DATA_PATH = 'Generated_dataset/testing_set_annotated.txt'


# Remove "task_desc"s whose "high_idx" is -2, -3, or -4.
# This is because the their task descriptions are the same, but the intents of
# the task is different depending on the location of the target and receptacle.
# In entity linking, the underlying intents of a task do not matter
def remove_duplicated_task_desc(input_file, output_file):
    output = open(output_file, "w")

    with open(input_file, "r") as file:
        for line in file:
            json_object = json.loads(line)

            if json_object["record_type"] == "task_desc":
                if json_object["high_idx"][0] == -1:
                    output.write(line)
            else:
                output.write(line)

    output.close()


# remove_duplicated_task_desc(TRAIN_DATA_PATH, NEW_TRAIN_DATA_PATH)
# remove_duplicated_task_desc(DEV_DATA_PATH, NEW_DEV_DATA_PATH)
# remove_duplicated_task_desc(TEST_DATA_PATH, NEW_TEST_DATA_PATH)


# --------------------------------------------------------------------------
# ------------------------ Create Annotated Dataset ------------------------
# --------------------------------------------------------------------------


# Read the gold label that is provided in the dataset,
# return the label that are processed by the nlp
# For example, input [ToiletPaper] -> nlp("toilet paper")
def get_label_embeddings(label_string, nlp):
    label = re.sub(r"([A-Z])", r" \1", label_string).split()
    label = ' '.join(label).lower()

    entity = nlp(label)
    return entity


# Read the text that is processed by the nlp, namely: nlp(text),
# Return textual mentions of entities in the text
def get_textual_entities(text):
    text.spans["all"] = []

    for chunk in text.noun_chunks:
        entity = None
        for token in chunk:
            if token.pos in [NOUN, PROPN]:
                start = token.i
                end = chunk.end
                entity = text[start:end]
                break
        # if the noun_chunk is not PRON (it, this, that)
        if entity:
            if not check_repetitive_mention(entity, text.spans["all"]) \
                    and not check_direction_related_noun(entity):
                text.spans["all"].append(entity)
                # print(entity.text)

    return text.spans["all"]


# Check if the pending_entity has already been mentioned in the text
# For example, in the sentence: "pick up the book and put the book on the table",
# "book" has been mentioned twice. To reduce matching efforts when annotating
# entities, only the first occurrence of the repetitive mentions of entity will be kept.
def check_repetitive_mention(pending_entity, entity_list):
    coref = False
    for entity in entity_list:
        sim = entity.root.similarity(pending_entity.root)
        if not pending_entity.root.has_vector:
            print(pending_entity.text, pending_entity.doc.text)
        if sim > 0.9:
            coref = True
    return coref


def check_direction_related_noun(entity):
    root_noun = entity.root
    related = False
    if root_noun.lower_ in ("right", "left", "middle", "end"):
        related = True
    return related


# Read the target/receptacle label, and the entity list of the text sentence
# Link the label with the textual mentions of entities
def link_label_with_textual_entity(label, entity_list, nlp):
    max_sim = 0
    max_entity = None

    if not label[0].has_vector:
        label.vector = nlp.vocab.get_vector(label[0].text, minn=4)

    for entity in entity_list:
        sim = label.similarity(entity)
        if sim > max_sim:
            max_sim = sim
            max_entity = entity

    return max_entity

# nlp = spacy.load("en_core_web_lg", disable=["ner"])
# bottle = nlp("Glassbottle")
# command = nlp("give me a bottle of water")
# list = get_textual_entities(command)
# link_label_with_textual_entity(bottle, list, nlp)


'''
Examples of data entry in the generated dataset

{
    "text": "put a credit card on a chair",
    "high_idx": [-1],
    "entities": [
        {
            "start_char": 6,
            "end_char": 17,
            "start_token": 2,
            "end_token": 4,
            "value": "credit card",  # textual mention of entity in the sentence
            "label": "CreditCard",   # the gold label provided in the dateset
            "role": "target",
        },
        {
            "start_char": 23,
            "end_char": 28,
            "start_token": 6,
            "end_token": 7,
            "value": "chair",
            "label": "ArmChair", 
            "role": "receptacle",
        }
    ]
}

{
    "text": "go to the table",
    "high_idx": [0]
    "entities": [],
}
'''


# Clean up the text command
def normalise_sentence(text_string):
    text_string = text_string.replace('\"', '')
    text_string = text_string.replace('/', '')
    text_string = text_string.replace('\n', '')
    # remove the period at the end of the sentence
    text_string = text_string.rstrip('.')
    text_string = text_string.lower()
    return text_string


# Return the task string of a data entry
def get_task_string(json_object):
    task_string = ''
    if json_object["record_type"] == "multi_desc":
        sub_tasks = []
        for task in json_object["desc"]:
            sub_task = normalise_sentence(task) + '.'
            sub_tasks.append(sub_task)
        task_string = ' '.join(sub_tasks)
    else:
        task_string = normalise_sentence(json_object["desc"][0])

    return task_string


# Determine the number of target and receptacle in a data entry
def determine_number_of_entities(json_object):
    """
    ---------------------------------------------------------
    record type                    target          receptacle
    ---------------------------------------------------------
    task_desc                        1                 1
    ---------------------------------------------------------
    Goto                             0                 0
    ---------------------------------------------------------
    Pickup                           1                 0
    ---------------------------------------------------------
    Put                              1                 1
    ---------------------------------------------------------
    Goto + pickup                    1                 0
    ---------------------------------------------------------
    Goto + pickup + goto             1                 0
    ---------------------------------------------------------
    Goto + pickup + goto + put       1                 1
    ---------------------------------------------------------
    pickup + goto                    1                 0
    ---------------------------------------------------------
    pickup + goto + put              1                 1
    ---------------------------------------------------------
    goto + put                       1                 1
    ---------------------------------------------------------
    """

    target_recep = None
    if json_object["record_type"] == "task_desc":
        target_recep = (1, 1)
    elif json_object["record_type"] == "high_desc":
        if json_object["high_idx"][0] in [0, 2]:
            target_recep = (0, 0)
        elif json_object["high_idx"][0] == 1:
            target_recep = (1, 0)
        elif json_object["high_idx"][0] == 3:
            target_recep = (1, 1)
    elif json_object["record_type"] == "multi_desc":
        if json_object["high_idx"] == [0, 1]:
            target_recep = (1, 0)
        elif json_object["high_idx"] == [0, 1, 2]:
            target_recep = (1, 0)
        elif json_object["high_idx"] == [0, 1, 2, 3]:
            target_recep = (1, 1)
        elif json_object["high_idx"] == [1, 2]:
            target_recep = (1, 0)
        elif json_object["high_idx"] == [1, 2, 3]:
            target_recep = (1, 1)
        elif json_object["high_idx"] == [2, 3]:
            target_recep = (1, 1)

    return target_recep


# Create a dictionary for an entity
def create_entity_dictionary(entity, label_string):
    entry = {"start_char": entity.start_char, "end_char": entity.end_char, "start_token": entity.start,
             "end_token": entity.end, "value": entity.text, "label": label_string, "role": None}
    return entry


# Create the annotated dataset
def generate_annotated_dataset(input_file, output_file, nlp):
    output_f = open(output_file, "w")

    with open(input_file, "r") as input_f:
        for line in input_f:
            new_entry = {}
            json_object = json.loads(line)
            text_string = get_task_string(json_object)

            for obj in json_object["scene_description"]:
                if 'object_type' in obj.keys():
                    if obj["object_type"] == 'simple':
                        target_string = obj["entityName"]
                        target = get_label_embeddings(target_string, nlp)
                    elif obj["object_type"] == 'receptacle':
                        recep_string = obj["entityName"]
                        receptacle = get_label_embeddings(recep_string, nlp)

            task_command = nlp(text_string)
            entity_list = get_textual_entities(task_command)

            new_entry["text"] = text_string
            new_entry["high_idx"] = json_object["high_idx"]
            new_entry["entities"] = []

            if len(entity_list) == 0:
                print(task_command, new_entry["high_idx"])
            target_recep = determine_number_of_entities(json_object)
            if target_recep[0] == 1:
                max_target = link_label_with_textual_entity(target, entity_list, nlp)
                target_entry = create_entity_dictionary(max_target, target_string)
                target_entry["role"] = "target"
                new_entry["entities"].append(target_entry)
            if target_recep[1] == 1:
                max_recep = link_label_with_textual_entity(receptacle, entity_list, nlp)
                recep_entry = create_entity_dictionary(max_recep, recep_string)
                recep_entry["role"] = "receptacle"
                new_entry["entities"].append(recep_entry)
            # print(new_entry)
            output_f.write(json.dumps(new_entry))
            output_f.write("\n")

        output_f.close()


# Print out the statistics of a given dataset
def produce_statistics(input_file):
    total_count = 0
    task_count = 0
    high_count = 0
    multi_count = 0
    with open(input_file, "r") as f:
        for line in f:
            total_count += 1
            json_object = json.loads(line)
            if json_object["record_type"] == "task_desc":
                task_count += 1
            elif json_object["record_type"] == "high_desc":
                high_count += 1
            elif json_object["record_type"] == "multi_desc":
                multi_count += 1

    print("Dataset:       ", input_file)
    print("Total tasks:   ", total_count)
    print("Task desc:     ", task_count)
    print("High desc:     ", high_count)
    print("Multi desc:    ", multi_count, "\n")


nlp = spacy.load("en_core_web_lg", disable=["ner"])

# generate_annotated_dataset(NEW_TRAIN_DATA_PATH, ANNOTATED_TRAIN_DATA_PATH, nlp)
# generate_annotated_dataset(NEW_TEST_DATA_PATH, ANNOTATED_TEST_DATA_PATH, nlp)
generate_annotated_dataset(NEW_DEV_DATA_PATH, ANNOTATED_DEV_DATA_PATH, nlp)

print("Completed")

# produce_statistics(NEW_TRAIN_DATA_PATH)
# produce_statistics(NEW_TEST_DATA_PATH)
# produce_statistics(NEW_DEV_DATA_PATH)
