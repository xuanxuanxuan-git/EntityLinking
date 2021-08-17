# EntityLinker

The multi-modal entity linker will identify textual entities in a sentence,
and link textual entities with visual entities. Visual entities are detected
by the existing [visual pipeline](https://github.com/stereolabs/zed-yolo). 

# speechToText

We use the [IBM Watson](https://cloud.ibm.com/catalog/services/speech-to-text) 
speech recognition service to perform verbal command recognition.


  
# IntentClassifier

This project uses RASA's DIET classifier for intent classification

The model that was trained with RASA's DIET was compared against a MultiLayer Perceptron.

The classifier uses both language and visual data to classify the intents behind a spoken command.
The visual data is first captured by an object detection system and converted into numerical visual features.
We use spacy's pretrained language model to generate features for the natural language text.

The project also includes a couple of custom components that extended the capabilities of the DIET classifier
