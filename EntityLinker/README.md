## Entity recognition - LSTM model
The LSTM model is responsible for entity recognition. 
The trained LSTM model can be found in the LSTM_model folder. 
The name of the model is called: <code>entity_model.pth</code>.
To use the model, please follow the steps below.

## Install spaCy and LSTM model related module

```angular2html

    python version = 3.7

    $ pip3 install -U pip setuptools wheel
    $ pip3 install -U spacy
    $ python -m spacy download en_core_web_trf (this model is used for dependency parsing)
    $ python -m spacy download en_core_web_lg (this model is used to generate word vectors)

    $ pip3 install pandas

    To check if pyTorch is installed:
    $ pip3 show torch

    If pytorch is not installed:
    $ pip3 install torch 
```

## Dataset for training the LSTM model
    
The datasets used for training the LSTM model can be 
    found in the folder <code>Data/Generated_dataset/lstm_dataset</code>.

```angular2html
    
    sentences.txt file: contains all sentences, one sentence per line.
    dep.txt file: each line contains dependency relation of each token.
    labels.text file: each line contains the role of each token.
    
```

## Generate the dataset for training the LSTM model

To re-generate the dataset, we need to run the following codes. Before this, we need to
change the <code>PARENT_FOLDER_PATH</code> variable in the generate_dataset.py to your own path
to the folder EntityLinker. Then uncomment codes in the main function.

```angular2html

    $ cd Data
    $ python build_labels.py     
    $ python generate_dataset.py

```

## Train the LSTM model

    
Open the entityLinker.ipynb file. It is recommended to upload the notebook and datasets 
generated in the previous step to Google Drive, and run the notebook on Google Colab.

To train the classifier, run the notebook up the section 'Save and load model'. After downloading
the model, copy the model and place it into the folder titled 'EntityLinker/LSTM_model'.
    


## Run the entity recognition, entity linking, and intent classification

### Run the entity recognition (LSTM model)
Step 0: change the paths
1. change SPEECH_RECOGNITION_PATH to your own path to the folder <i>speechToText</i>.
2. change PARENT_FOLDER_PATH to your own path to the folder <i>EntityLinker</i>.
3. change INTENT_CLASSIFICATION to your own path to the folder <i>IntentClassification/rasa_custom</i>.
4. make sure that the trained LSTM model is placed in the folder <i>LSTM_model</i>, and is titled as <i>entity_model.pth</i>.

Step 1: perform the entity recognition
```angular2html
    
    $ cd LSTM_model
    $ python entity_recognition.py
    Enter 1 to type a command.
    
    
    To test the LSTM model, enter the following sentences
    
        INPUT:
        "move the purple pillow from the couch to the arm chair"
        
        OUTPUT: 
        target: ['pillow']
        receptacle: ['arm', 'chair']
    
        INPUT:
        "pick up the book and put it on the counter"
    
        OUTPUT: 
        target: ['book']
        receptacle: ['counter']
    
        INPUT:
        "hang a right at the wooden dresser and walk to the brown chair ahead"

        OUTPUT: 
        target: []
        receptacle: []

        INPUT: "Pick the book from the corner of the shelf"

        OUTPUT:
        target: ['book']
        receptacle: []
    
    To exit the inference: enter Ctrl+C, then press enter

```

### Enable the speech recognition service

Step 0.1: install additional packages
```angular2html
    $ pip3 install pyaudio
    $ pip3 install websocket-client==0.56.0
```

Step 0.2: create a file named <code>speech.cfg</code> in the folder <i>speechToText</i>, copy the information from 
<i>speech.cfg.example</i> to this file. Replace the <i>apikey</i> and <i>instance_id</i>
with the authentication details provided in the last section of the handover report.

Step 1: issue a verbal command
```angular2html
    $ python entity_recognition.py
    Enter 2, wait for the instruction '* please speak', and then give a verbal command.
```

### Run the visual and textual entity linking

Step 0.1: turn on the visual pipeline. Change the receiver's ip address to the current device.

Step 0.2: install all the packages needed for the intent classification. And do the following commands.

```angular2html
    In the IntentClassification/rasa_custom/rasa_single_instance_tester.py, change the 
    rootpath to your own path to the folder IntentClassification.

    Then start the RASA server in a separate terminal and run the following command:
    $ rasa run --enable-api -m models/20201123-111915.tar.gz
```

Step 1: issue a command and perform the intent classification
```angular2html
    $ python entity_recognition.py
    Enter 1 or 2, then enter a command. The terminal will display the recognised target and receptacle 
    using the LSTM model, the visual mentions of same entities, and the predicted intents
    from the multi-modal intent classifier.
```