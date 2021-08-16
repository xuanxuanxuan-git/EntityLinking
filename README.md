## ChangeLog (16 August 2021)

1. Restructure the folder
2. Add a new folder: EntityLinker
    * parsing.py
3. Update the speech_recognition.py

### Details (Incomplete)


## ChangeLog (09 August 2021)

1. Add a new folder: speechToText
   * <code>recognition.py</code> 
   * <code>textToSpeech.py</code>
   * <code>background_listening.py</code>
2. Add an extra function <code>speech_to_rasa</code> in <code>rasa_single_instance_tester.py</code>

### Details
- recognition.py will listen to the voice command, transcribe the audio data to the text.
- textToSpeech.py will read out text input
   * to run the text-to-speech, install pyttsx3 through: pip3 install pyttsx3
   * to test the voice of the engine, uncomment the last two lines in textToSpeech.py
   * the voice used is <b>system specific</b>. The code will first detect the os and set the voice available in that os.
- background_listening.py allows continuous background listening, and detecting keywords. The main speech-to-text 
  recognition will be activated when wake words such as "hi", "hello" are heard. Then the program will listen for the 
  voice command, transcribe it and invoke <code>post_to_rasa</code> to do intent classification. 
- <code>speech_to_rasa</code> in rasa_single_instance_tester.py can activate the speech-to-text recognition 
  and feed the transcribed command to rasa for intent classification.
  
  
# IntentClassifier

This project uses RASA's DIET classifier for intent classification

The model that was trained with RASA's DIET was compared against a MultiLayer Perceptron.

The classifier uses both language and visual data to classify the intents behind a spoken command.
The visual data is first captured by an object detection system and converted into numerical visual features.
We use spacy's pretrained language model to generate features for the natural language text.

The project also includes a couple of custom components that extended the capabilities of the DIET classifier
