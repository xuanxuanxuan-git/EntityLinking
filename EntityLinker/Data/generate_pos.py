import linecache
import spacy
import json

TRAIN_SENTENCES_FILE_PATH = "Generated_dataset/baseline_dataset/sentences_train.txt"
DEV_SENTENCES_FILE_PATH = "Generated_dataset/baseline_dataset/sentences_dev.txt"
TEST_SENTENCES_FILE_PATH = "Generated_dataset/baseline_dataset/sentences_test.txt"

TRAIN_LABELS_FILE_PATH = "Generated_dataset/baseline_dataset/labels_train.txt"
DEV_LABELS_FILE_PATH = "Generated_dataset/baseline_dataset/labels_dev.txt"
TEST_LABELS_FILE_PATH = "Generated_dataset/baseline_dataset/labels_test.txt"

CRF_TRAIN = "Generated_dataset/crf_dataset/crf_train.txt"
CRF_DEV = "Generated_dataset/crf_dataset/crf_dev.txt"
CRF_TEST = "Generated_dataset/crf_dataset/crf_test.txt"


# get crf features for a single sentence
def get_pos(nlp, sentence, labels):
    sent = nlp(sentence)
    label = labels.split(' ')
    features = []

    i = 0
    for token in sent:
        result = (token.text, token.pos_, label[i])
        features.append(result)
        i += 1
    return features


def get_crf_dataset(nlp, sentence_file, label_file, output_file):
    output = open(output_file, "w")
    index = 1

    with open(sentence_file, "r") as sent_f:
        for line in sent_f:
            sentence = line.rstrip()
            label = linecache.getline(label_file, index).rstrip()

            feature = get_pos(nlp, sentence, label)
            json.dump(feature, output)
            output.write('\n')
            index += 1
    output.close()


nlp = spacy.load("en_core_web_lg", disable=["ner"])
# get_crf_dataset(nlp, TRAIN_SENTENCES_FILE_PATH, TRAIN_LABELS_FILE_PATH, CRF_TRAIN)
# get_crf_dataset(nlp, DEV_SENTENCES_FILE_PATH, DEV_LABELS_FILE_PATH, CRF_DEV)
# get_crf_dataset(nlp, TEST_SENTENCES_FILE_PATH, TEST_LABELS_FILE_PATH, CRF_TEST)
print("complete")