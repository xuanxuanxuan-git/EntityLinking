### Intents to Reachy's actions

We use [Reachy](https://docs.pollen-robotics.com/) to perform the pickup and put actions.
You can look at their [manual](https://docs.pollen-robotics.com/sdk/getting-started/introduction/) 
for more information.

### Codes
1. <code>reachy_start.py</code>

To start Reachy, first you need to install the package by:
```angular2html
pip3 install reachy-sdk
```
The functions in this file are directly invoked by the main function in entity_recognition.py. 
You can also run this file to reset the coordinates of the robot or get the current pose of the robot. 


2. <code>intent_to_reachy.py</code>

This file maps the intent inference from the intent classifier to a sequence of actions by Reachy.

3. <code>EntityLinker/LSTM_model/coordinate_transformation.py</code>

This file transforms the vectors in the camera's coordinate system to the vectors 
in the robot's system. To run the program, we need to install by:
```angular2html
pip3 install scipy
```