# Humanoid robotic arm control through verbal commands.

This project allows the user to manipulate the robotic arm to perform various reaching and grasping 
movements. The pipeline would allow the control through verbal commands without any or with limited 
training by the user. The intents behind the natural language command can be understood by the 
intent classifier. The objects involved in the command can be recognised by the entity linker. 
The positional information about the objects can be detected by the visual pipeline. Then the predicted 
intents along with the positional information of the relevant objects are used to generate a sequence 
of actions that can be performed by the robotic arm - Reachy.

The pipeline consists of the following components.

<hr>

## EntityLinker (Yueqing Xuan, 2021)

The multi-modal entity linker is capable of recognising the target and receptacle entities 
within a command. It can also perform the visual and textual entity linking, which link textual
mentions of entities with visual mentions of entities. Visual entities are detected
by the existing [visual pipeline](https://github.com/stereolabs/zed-yolo). With the visual
and textual co-reference links, the entity linker can get the visual information of the target
and the receptacle, and pass the information to multi-modal intent classifier for intent 
classification.

## speechToText

We use the [IBM Watson](https://cloud.ibm.com/catalog/services/speech-to-text) 
speech recognition service to transcribe the verbal commands to textual commands, so that they 
can be used for entity recognition and intent classification.

We also use the [Porcupine Wake Word Engine](https://picovoice.ai/platform/porcupine/) 
to detect the wake word <i>Hey Reachy</i>.

The speech recognition and entity linking are the first step in our entire pipeline. We first 
transcribe verbal commands to textual commands, and identify the target object and the receptacle 
object involved in the commands.
  
## VisualPipeline (Zafar Faraz, 2020)

The visual pipeline is equipped with a stereo camera and a GPU. It is capable of detecting objects 
in nearly real time. Currently it only supports 80 categories of objects listed in the COCO dataset. 
It publishes its detection results to a server by sending UDP messages. 

The codes related to the visual pipeline can be found in this [repo](https://github.com/stereolabs/zed-yolo).


## IntentClassifier (Karun Matthew, 2020)

This project uses RASA's DIET classifier for multi-modal intent classification

The model that was trained with RASA's DIET was compared against a MultiLayer Perceptron.

The classifier uses both language and visual data to classify the intents behind a spoken command.
The visual data is first captured by an object detection system and converted into numerical visual features.
We use spacy's pretrained language model to generate features for the natural language text.

The project also includes a couple of custom components that extended the capabilities of the DIET classifier

The codes related to the multi-modal intent classifier can be found in this [repo](https://github.com/aditya30887/IntentClassifier/tree/master/IntentClassification).

## reachy (Yueqing Xuan, 2021)

We use [Reachy](https://docs.pollen-robotics.com/) to perform the pickup and put actions. This is the last 
step in our entire pipeline. It uses the intent prediction results from the intent classifier and 
the positional information from the visual pipeline (after coordinate transformation) to generate 
a sequence of actions to be performed by Reachy.


<hr>

## Run the entire pipeline

The main function to start the entire pipeline can be found in <code>EntityLinker/LSTM_model/entity_recognition.py</code>.
To run the program, please follow the instruction in <code>EntityLinker/README.md</code>.