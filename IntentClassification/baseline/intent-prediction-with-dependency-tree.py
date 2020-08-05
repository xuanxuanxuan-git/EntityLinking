# This solution does not have any training phase as it uses a set of generic
# rules to perform the intent classification


import spacy
from sklearn.metrics import precision_recall_fscore_support

# INPUT_FILE = "outfile1"

# Provide the path of the input file
# TODO Currently it is tab delimited, it may change to JSON in the near future
INPUT_FILE = '../data-train/outfile_pickup_simple_train_unique'
READ = "r"
# any two documents whose spacy's document similarity score is greater than
# this value is considered as a match
SIMILARITY_THRESHOLD = 0.6

# load the spacy's english model
nlp = spacy.load("en_core_web_lg")

# load the file that contains the intents to classify
file = open(INPUT_FILE, READ)

# domain specific actions are listed here
# this is to be manually compiled
actions = {'pick': 'GRASP',
           'move': 'MOVE_FORWARD',
           'take': 'GRASP',
           'go': 'MOVE_FORWARD',
           'place': 'RELEASE',
           'put': 'RELEASE',
           }

complex_actions = {
    'take to': 'TRANSPORT',
    'turn left': 'MOVE_LEFT',
    'move left': 'MOVE_LEFT',
    'move right': 'MOVE_RIGHT',
    'turn right': 'MOVE_RIGHT',
    'turn around': 'MOVE_BACK',
    'turn on': 'NOT_SUPPORTED'
}

# the all_actions dict will store both kinds of actions
# TODO In the future distinction between actions and complex actions may
#      be removed
all_actions = {}
all_actions.update(actions)
all_actions.update(complex_actions)


# returns more details for a particular identified intent
# it returns the words that are connected to the passed verb in the
# dependency tree. The connected words usually will be adverbs
def get_intent_details(doc, verb, verb_index):
    adverb = ''
    adposition = ''
    for token in doc:
        if token.dep_ == 'advmod' and token.head.text == verb and \
                token.head.i == verb_index and adverb == '':
            print('advmod :::::::::::::::::: ', token.text)
            adverb = token.text
        elif token.pos_ == 'ADP' and token.head.text == verb and \
                token.head.i == verb_index and adposition == '':
            print('ADP :::::::::::::::::: ', token.text)
            adposition = token.text
        elif token.pos_ == 'ADV' and token.head.text == verb and \
                token.head.i == verb_index and adverb == '':
            print('ADV :::::::::::::::::: ', token.text)
            adverb = token.text

    return adverb, adposition


# pick the best intend from the list of possible intends
# if the verb along with the adverb directly matches the keys of the complex
# action then the corresponding action is returned as intent
# if there is no complex action match, we check for normal action match.
# if there is no complex and normal action match, then we use spacy's document
# representation that uses word embeddings of each contained token to find
# the similarity between the passed verb phrase and the keys of the all_actions
# dict, and return the one with the highest similarity score
def get_intent_class(verb, adverb, adposition):
    verb = verb.strip().lower()
    verb_with_adverb = verb + ' ' + adverb.strip().lower()
    verb_with_adposition = verb + ' ' + adposition.strip().lower()

    if verb_with_adverb in complex_actions:
        return complex_actions[verb_with_adverb]
    elif verb_with_adposition in complex_actions:
        return complex_actions[verb_with_adposition]
    elif verb in actions:
        return actions[verb]
    else:
        return get_most_similar_action(verb)


# get spacy similarity score between words
def get_spacy_similarity_score(word1, word2):
    doc1 = nlp(word1)
    doc2 = nlp(word2)
    return doc1.similarity(doc2)


# returns the similarity score and the action that the extracted
# verb and modifiers are most similar to
def get_most_similar_action(verb):
    most_similar_action = 'NOT_SUPPORTED'
    most_similar_action_score = -1
    for action in all_actions.keys():
        score = get_spacy_similarity_score(action, verb)
        if score > most_similar_action_score:
            most_similar_action = action
            most_similar_action_score = score

    print("SIM SCORE :::: ", most_similar_action_score, '  ',
          most_similar_action)
    if most_similar_action_score > 0.6:
        return all_actions[most_similar_action]
    else:
        return 'NOT_SUPPORTED'


# returns the main intent for the passed command sentence
def get_intent_for_command(line):
    doc = nlp(line)
    root_verbs = []
    root_verb_indices = []
    verbs = []
    verb_indices = []

    for token in doc:
        # print(token.text, ' ', token.pos_, ' ',
        # token.dep_, ' ', token.head.text)
        if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
            root_verbs.append(token.text)
            root_verb_indices.append(token.i)
        elif token.pos_ == 'VERB':
            verbs.append(token.text)
            verb_indices.append(token.i)
    # print(line)
    # print(root_verbs)
    # print(root_verb_indices)
    # print(verbs)
    # print(verb_indices)
    main_intent = ''
    for root_verb, root_verb_index in zip(root_verbs, root_verb_indices):
        root_adverb, root_adposition = get_intent_details(doc, root_verb,
                                                          root_verb_index)
        print('MAIN INTENT --- ', root_verb, ' ', root_adverb, ' ',
              root_adposition)
        if main_intent == '':
            main_intent = get_intent_class(root_verb, root_adverb,
                                           root_adposition)
        print('MAIN INTENT CLASS --- ', main_intent)
    if len(root_verbs) > 1:
        print('Compound sentence')
    if len(verbs) > 0:
        for verb, verb_index in zip(verbs, verb_indices):
            adverb, adposition = get_intent_details(doc, verb, verb_index)
            other_intent = get_intent_class(verb, adverb, adposition)

            if len(root_verbs) == 0 and main_intent == '':
                # if the root word was identified as a noun
                main_intent = other_intent

            print('OTHER INTENT --- ', verb, ' ', adverb, ' ', adposition)
            print('OTHER INTENT CLASS --- ', other_intent)
    return main_intent


# checks if the predicted intent for a passed command matches the actual intent
# the command and the actual intent are passed in as input in a tab separated
# format
def classify_intent_for_test_record(data):
    print(data)
    cols = data.split('\t')
    line = cols[0]
    tag = cols[1]

    main_intent = get_intent_for_command(line)

    print("ACTUAL CLASS :: ", tag)
    if tag == main_intent:
        print('MATCH\n')
        return True, main_intent, tag
    # temporarily not considering unsupported tags
    # elif tag == 'NOT_SUPPORTED':
    #    return True
    else:
        print('NOT MATCH\n')
        return False, main_intent, tag


def classify_intent(is_limit=False, limit=2000):
    count = 0.0
    correct = 0.0

    predicted_tags = []
    actual_tags = []

    for line in file:
        count += 1
        if is_limit and count > limit:
            break
        line = line.strip()
        result, predicted_tag, actual_tag = classify_intent_for_test_record(
            line)
        predicted_tags.append(predicted_tag.strip())
        actual_tags.append(actual_tag.strip())

        if result:
            correct += 1

    print(precision_recall_fscore_support(actual_tags, predicted_tags,
                                          average='macro'))
    print(precision_recall_fscore_support(actual_tags, predicted_tags,
                                          average=None,
                                          labels=['MOVE_LEFT', 'MOVE_RIGHT',
                                                  'MOVE_BACK', 'MOVE_FORWARD',
                                                  'RELEASE', 'GRASP',
                                                  'TRANSPORT',
                                                  'NOT_SUPPORTED']))
    print('Accuracy :: ', correct / count)
    print('Correct  :: ', correct)
    print('Total    :: ', count)


classify_intent()
# get_intent_for_command('take a step to your left')

file.close()