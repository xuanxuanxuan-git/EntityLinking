## Run the entire pipeline

1. Installation

To run the entire pipeline, first you need to install necessary packages for relevant components. 
Please make sure that you are using python version 3.6 or python version 3.7. It is recommended 
that you use virtual environments to run the project to avoid potential conflicts.


Requirements for speech recognition tools:
```angular2html
$ pip install pvporcupinedemo                (for wake word engine)
$ pip install websocket-client==0.56.0       (for speech recognition)
```

Requirements for the entity linker:
```angular2html
$ pip3 install -U spacy
$ python -m spacy download en_core_web_trf (this model is used for dependency parsing)
$ python -m spacy download en_core_web_lg (this model is used to generate word vectors)

$ pip3 install pandas

To check if pyTorch is installed:
$ pip3 show torch

If pytorch is not installed:
$ pip3 install torch 
```


Requirements for the intent classification:
```angular2html
Please note that we use a separate virtual environment to run the intent classifier. The reasons 
are described in the onenote (RA week 3 -> known issues -> point 5).

$ python3.7 -m venv ./venv                    (make sure you do this inside the IntentClassification folder)
$ source ./venv/bin/activate
$ pip install -U pip
$ pip install rasa==1.10.12
 OR
$ pip3 install rasa==1.10.12
$ pip3 install rasa[full]==1.10.12            (run this to make sure you install all dependencies)

Dependencies for Spacy

$ pip3 install rasa[spacy]==1.10.12
$ python3 -m spacy download en_core_web_lg
$ python3 -m spacy link en_core_web_lg en
```

Requirements for Reachy's execution
```angular2html
$ pip install reachy-sdk
$ pip install scripy
```

Requirements for the visual pipeline
```angular2html
We run the visual pipeline in the GPU. Please refer to the OneNote (RA week 3 -> Run the entire 
pipeline step by step) to see how to connect the GPU and start the visual pipeline.
```

2. Create the wake word engine model

```angular2html
Please read through the readme.md file in speechToText folder for creating the wake word model.
Please put the model in speechToText/wake_word/hey_reachy_{os} folder.
```

3. Change folder paths

a) In <code>LSTM_model/entity_recognition.py</code>

```angular2html
Replace the SPEECH_RECOGNITION_PATH, PARENT_FOLDER_PATH, INTENT_CLASSIFICATION, REACHY_PATH,
INTENT_INPUT_FILE, INTENT_OUTPUT_FILE variables with the paths in your system accordingly.
```

b) In <code>speechToText/wake_word/wake_word_engine.py</code>
```angular2html
Replace the KEYWORD_PATH variable with the path to the wake word model in your system.
```

c) In <code>IntentClassification/rasa_custom/rasa_single_instance_tester.py</code>
```angular2html
Replace rootpath, intent_input_path, intent_prediction_path variables accordingly.
```

4. Fill in the camera setup information
```angular2html
In line 55-57 in entity_recognition.py, fill in the direction of the camera, the translation 
vector, as well as the angle of the camera. Please refer to the onenote (RA week 3 -> 
coordinate transformation) for more information about the calculation.
```

5. Start the intent classifier
```angular2html
Open a new terminal, activate the virtual environment for the intent classifier. 

Run the following commands:
$ Export PYTHONPATH={path/to/rasa_custom}/:PYTHONPATH
$ Export PYTHONPATH={path/to/IntentClassification}/:PYTHONPATH
$ cd {path/to/IntentClassification}
$ rasa run --enable-api -m models/20201123-111915.tar.gz

Open another new terminal, activate the virtual environment again and export relevant paths.
Then run the following commands:
$ cd {path/to/rasa_custom}
$ python rasa_single_instance_tester.py
```

6. Start the robot
```angular2html
Please refer to the OneNote (RA week 3 -> Run the entire pipeline step by step) to see 
how to connect and start the robot.
```

7.Start the entire pipeline
```angular2html
The entire pipeline is started by executing the main function in the LSTM_model/entity_recognition.py.
You need to run the following commands:

$ cd {path/to/LSTM_model}
$ python entity_recognition.py

Then it will be prompted to enter the mode:
Enter 1 to type the command, enter 2 to start wake word detection and then give a verbal command.
Enter 'Ctrl+C' to quit.

You need to re-start the rasa_single_instance_tester.py (in step 5) everytime you re-run 
the pipeline.
```



<br><hr>

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

Step 1: perform the entity recognition
```angular2html
    
    $ cd LSTM_model
    $ python entity_recognition.py
    
    
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
