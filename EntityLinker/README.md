## LSTM classifier

The LSTM classifier can be found in the LSTM_model folder. 
The name of the classifier is called: <code>entity_model.pth</code>.
To use the model, please follow the steps below.

## Install spaCy and classifier related module

```angular2html

    $ pip install -U pip setuptools wheel
    $ pip install -U spacy
    $ python -m spacy download en_core_web_trf (optional, this model is used for generating dataset)
    $ python -m spacy download en_core_web_lg

    $ pip install pandas
```

## Dataset for training the classifier
    
The datasets used for training the LSTM classifier can be 
    found in the folder <code>Data/Generated_dataset/baseline_dataset</code>.

```angular2html
    
    sentences.txt file: contains all sentences, one sentence per line.
    dep.txt file: each line contains dependency relation of each token.
    labels.text file: each line contains the role of each token.
    
```

## Generate the dataset for model training 

```angular2html

    $ cd Data
    $ python build_labels.py     (this will generate the labels file)
    $ python generate_dataset.py (this will generate the sentences file and dependency relation file)

```

## Train the LSTM classifier

```angular2html

    $ cd LSTM_model
    
    Open the entityLinker.ipynb file. It is recommended to upload the notebook and datasets 
    generated in the previous step to Google Drive, and run the notebook on Google Colab.

    To train the classifier, run the notebook up the section 'Save and load model'.
```


## Run the trained LSTM classifier

```angular2html
    
    $ cd LSTM_model
    $ python entity_recognition.py
    
    To test the classifier, enter the following sentences
    
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
        "hang a right at the wooden dresser and walk to the brown"

        OUTPUT: 
        target: []
        receptacle: []

        INPUT: "Pick the book from the corner of the shelf"

        OUTPUT:
        target: ['book']
        receptacle: []

```