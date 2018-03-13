# Tracking Head Motion to Control Computer Events
Description of all the files included in our code submission

## Android App

All the code for the Android app created in Unity is in the Data_App folder.

The C# scripts that contain the sensor collection, button functionality, and client code is under Assets/Scripts.

### ResetButton.cs
Contains basic functionality of reset button for counter (used to assist data collector)

### StartButton.cs
Contains functionality for start/stop button and toggles. Stores data in text file for data collection app and sends necessary data over established socket connection for real-time app.


## Classification

All the Python classification files are in the PyScripts folder.

### assistedServer.py
Server for "assisted live-streaming." Rather than using an overlapping, uniform frame approach, this server will receive sensor data until the client tells it to stop. Then, it will use the classifer from classify2.py to label the data, and trigger a computer event.

### server.py
Server for true real-time streaming. Continuously receives sensor data from client, and segments data with overlap to ensure that the action of interest is not missed. It will then use classifer from classify2.py to label the data, and trigger a computer event.

### classify2.py
Contains the code for training the classifier. Functions support two ways of training: (1) separate files into uniform overlapping frames to use for training; (2) use raw files (each with single isolated tilt) for training

### metrics.py
Contains data structure for the different features of importance

### window.py
Contains code specific to the window method, i.e. the uniform, overlapping frames method.

### utils.py
Contains general utility code, as well as code that was used earlier in the process for testing and feature extraction

