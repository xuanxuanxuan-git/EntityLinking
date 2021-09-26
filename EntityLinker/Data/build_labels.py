# Yueqing Xuan
# 1075355

"""
We store date in two different files:
    a sentence.txt file containing the sentences (one per line),
    a dep.txt file containing the dependency relation of each token to its head
    and a labels.txt containing the labels.

In sentences.txt file:
    Put a credit card on a chair

In dep.txt file:
    ROOT det compound dobj prep det pobj

In labels.txt file
    O O B-TARGET I-TARGET O O B-RECEPTACLE

B-TARGET represents the beginning of a TARGET entity,
if the entity has more than one token, subsequent tags are represented as I-TARGET
"""
import json
import spacy
import linecache
from generate_dataset import ANNOTATED_TRAIN_DATA_PATH, ANNOTATED_DEV_DATA_PATH, ANNOTATED_TEST_DATA_PATH


TRAIN_SENTENCES_FILE_PATH = "Generated_dataset/baseline_dataset/sentences_train.txt"
DEV_SENTENCES_FILE_PATH = "Generated_dataset/baseline_dataset/sentences_dev.txt"
TEST_SENTENCES_FILE_PATH = "Generated_dataset/baseline_dataset/sentences_test.txt"

TRAIN_LABELS_FILE_PATH = "Generated_dataset/baseline_dataset/labels_train.txt"
DEV_LABELS_FILE_PATH = "Generated_dataset/baseline_dataset/labels_dev.txt"
TEST_LABELS_FILE_PATH = "Generated_dataset/baseline_dataset/labels_test.txt"

TRAIN_DEP_FILE_PATH = "Generated_dataset/baseline_dataset/dep_train.txt"
DEV_DEP_FILE_PATH = "Generated_dataset/baseline_dataset/dep_dev.txt"
TEST_DEP_FILE_PATH = "Generated_dataset/baseline_dataset/dep_test.txt"


OUTSIDE = 'O'
B_TARGET = 'B-TARGET'
I_TARGET = 'I-TARGET'
B_RECEPTACLE = 'B-RECEPTACLE'
I_RECEPTACLE = 'I-RECEPTACLE'

TARGET = 'target'
RECEPTACLE = 'receptacle'


# Create the sentences.txt file
# Each line contains tokens of a sentence. Tokens are seperated by a space
def build_sentences_and_dep_file(input_file, output_token_file, output_dep_file, nlp_trf):

    output_token = open(output_token_file, "w")
    output_dep = open(output_dep_file, "w")

    with open(input_file, "r") as input_f:
        for line in input_f:
            json_object = json.loads(line)
            text_string = json_object["text"]
            text = nlp_trf(text_string)

            token_list = [t.text for t in text]
            token_string = ' '.join(token_list)

            dep_list = [t.dep_ for t in text]
            dep_string = ' '.join(dep_list)

            output_token.write(token_string)
            output_token.write("\n")

            output_dep.write(dep_string)
            output_dep.write("\n")

    output_token.close()
    output_dep.close()


# Create labels.txt file
def build_labels_file(input_file, sentences_file, output_file):
    output = open(output_file, "w")
    # the line numbers in the linecache module start with 1
    index = 1

    with open(input_file, "r") as input_f:
        labels = []
        for line in input_f:
            json_object = json.loads(line)

            token_list = linecache.getline(sentences_file, index).rstrip().split()

            labels = [OUTSIDE] * len(token_list)
            index += 1

            for entity in json_object["entities"]:
                # if there are no target or receptacle in the sentence
                # the label list for this sentence remains the same
                if entity:
                    length = entity["end_token"] - entity["start_token"]
                    if entity["role"] == TARGET:
                        labels[entity["start_token"]] = B_TARGET
                        if length > 1:
                            labels[(entity["start_token"]+1): entity["end_token"]] = [I_TARGET]*(length-1)
                    elif entity["role"] == RECEPTACLE:
                        labels[entity["start_token"]] = B_RECEPTACLE
                        if length > 1:
                            labels[(entity["start_token"]+1): entity["end_token"]] = [I_RECEPTACLE]*(length-1)

            labels_string = ' '. join(labels)

            output.write(labels_string)
            output.write("\n")

    output.close()


# nlp_trf = spacy.load("en_core_web_trf", disable=["ner"])
# build_sentences_and_dep_file(ANNOTATED_TRAIN_DATA_PATH, TRAIN_SENTENCES_FILE_PATH, TRAIN_DEP_FILE_PATH, nlp_trf)
# build_sentences_and_dep_file(ANNOTATED_DEV_DATA_PATH, DEV_SENTENCES_FILE_PATH, DEV_DEP_FILE_PATH, nlp_trf)
# build_sentences_and_dep_file(ANNOTATED_TEST_DATA_PATH, TEST_SENTENCES_FILE_PATH, TEST_DEP_FILE_PATH, nlp_trf)

#
build_labels_file(ANNOTATED_TRAIN_DATA_PATH, TRAIN_SENTENCES_FILE_PATH, TRAIN_LABELS_FILE_PATH)
build_labels_file(ANNOTATED_DEV_DATA_PATH, DEV_SENTENCES_FILE_PATH, DEV_LABELS_FILE_PATH)
build_labels_file(ANNOTATED_TEST_DATA_PATH, TEST_SENTENCES_FILE_PATH, TEST_LABELS_FILE_PATH)

print("completed")
