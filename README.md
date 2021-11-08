# EntityLinker

The multi-modal entity linker is capable of recognising the target and receptacle entities 
within a command. It can also perform the visual and textual entity linking, which link textual
mentions of entities with visual mentions of entities. Visual entities are detected
by the existing [visual pipeline](https://github.com/stereolabs/zed-yolo). With the visual
and textual co-reference links, the entity linker can get the visual information of the target
and the receptacle, and pass the information to multi-modal intent classifier for intent 
classification.

# speechToText

We use the [IBM Watson](https://cloud.ibm.com/catalog/services/speech-to-text) 
speech recognition service to transcribe the verbal commands to textual commands, so that they 
can be used for entity recognition.


  
# IntentClassifier

This project uses RASA's DIET classifier for intent classification

The model that was trained with RASA's DIET was compared against a MultiLayer Perceptron.

The classifier uses both language and visual data to classify the intents behind a spoken command.
The visual data is first captured by an object detection system and converted into numerical visual features.
We use spacy's pretrained language model to generate features for the natural language text.

The project also includes a couple of custom components that extended the capabilities of the DIET classifier
