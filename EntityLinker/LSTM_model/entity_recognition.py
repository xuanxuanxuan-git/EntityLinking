import spacy
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# from entityLinker import collect_dep_relations, create_dep_encoding, get_dep_vector

MODEL_PATH = 'entity_model.pth'
TRAIN_DEP_FILE_PATH = "../Data/Generated_dataset/baseline_dataset/dep_train.txt"
DEV_DEP_FILE_PATH = "../Data/Generated_dataset/baseline_dataset/dep_dev.txt"
TEST_DEP_FILE_PATH = "../Data/Generated_dataset/baseline_dataset/dep_test.txt"

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
        print("Error: Dependency relation not found")
        return None


# collect the dependency relation set, and represent each dep relation using
# one-hot encoding
dep_set = collect_dep_relations({TRAIN_DEP_FILE_PATH, DEV_DEP_FILE_PATH, TEST_DEP_FILE_PATH})
dep_df = create_dep_encoding(dep_set)


# ----------------------------------------------------------------------------- #


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

        # print("input packed shape:", input_packed.batch_sizes)
        lstm_out_packed, _ = self.lstm(input_packed)
        # print("output packed shape:", lstm_out_packed.batch_sizes)

        # lstm_out_padded shape: batch size x batch_max_len x lstm_hidden_dim
        lstm_out_padded, output_lengths = pad_packed_sequence(lstm_out_packed,
                                                              batch_first=True)

        # label_out size: batch_size x batch_max_len x roleset_size
        label_out = self.cls_layer(lstm_out_padded)

        return label_out


def idx_to_label(preds, role_to_idx):
    preds_string = []
    for result in preds:
        role = list(role_to_idx.keys())[list(role_to_idx.values()).index(result)]
        preds_string.append(role)
    return preds_string


def process_sentence(nlp, sentence_string):
    sentence_nlp = nlp(sentence_string.lower())

    num_token = len(sentence_nlp)
    tensor_shape = (num_token, feature_dimension)
    tokens_features_t = torch.zeros(tensor_shape)

    i = 0
    for token in sentence_nlp:
        dep = get_dep_vector(dep_df, token.dep_)
        vector = token.vector
        feature = np.append(vector, dep)
        feature_tensor = torch.from_numpy(feature).float()
        tokens_features_t[i] = feature_tensor
        i += 1

    return tokens_features_t


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


def command_to_entities(sentence):
    sentence_nlp = nlp(sentence.lower())

    result = predict(model_load, nlp, sentence, role_to_idx)
    target, receptacle = get_target_and_recep(sentence_nlp, result)

    print("\nThe command you entered is: ", sentence)
    print("Predictions of each word:  ", result)
    print("target:     {} \nreceptacle: {}\n".format(target, receptacle))


if __name__ == '__main__':
    # load the saved model
    model_load = EntityLinker(feature_dim=feature_dimension,
                              hidden_dim=hidden_dimension,
                              roleset_size=roleset_size)
    model_load.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    nlp = spacy.load("en_core_web_lg", disable=["ner"])

    # ------------- enter a sentence from the console -------------------- #

    while True:
        try:
            sentence = input("Enter a command: ")
            command_to_entities(sentence)
        except KeyboardInterrupt:
            print("\nQuit!")
            break

    # -------------- or manually type a sentence ------------------------ #

    # sentence = "move the pillow to the chair and turn to the couch"
    # command_to_entities(sentence)
