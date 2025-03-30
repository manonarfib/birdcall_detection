# Detection and identification of bird calls


## Description
This git repository is made to detect and recognize birds in sound files. It is part of the project of CentraleSupélec's S6.10.06 students.


## Installation
The required Python modules are listed in the 'requirement.txt' file. Beware that CUDA necesitates a specific installation and may cause compatibility issues.


## Usage of the application
In order to use the application, you must first ensure that all the requirements specified in the requirement.txt file are installed, and that Java 8 or more is available on your computer.
When opened, the application will prompt you with a file chooser window. You must then select the git project folder in order to continue.
This application works wwith drag and drop. To use it, you just have to drop the files you want to test the model on in the window.

## Usage
The 'creation_datasets' folder can be used to create the datasets to train and test the AI

To use the AI :
put in the input folder a csv file with a column audio_id containing the list of the link to the file you want to test the model on
...

no extension on the audio_id column
Note : the csv can have multiple columns, but only the audio_id is considered

To make the training :
download and preprocess the dataset : in the terminal, put yourself in the "birdcall-detection" folder. Then type "make prepare"
Wait until the download ends. Then you can launch the training with "make train" (still in the terminal).
If you have not enough RAM, you'd better train models one by one. To do so, write (still in the terminal of the "birdcall-detection" folder) :
python train -m 1
python train -m 2
python train -m 3
python train -m 4

Note : If you want to make your own dataset, in the train/data_training folder, put an "audio_files" folder and a csv with the adequate format :
cnt,en,id,length
**,class1,audio1.mp3,**
**,class1,audio2.mp3,**
...
**,class2,audio4.wav,**
...

where audio.* is just the name of the corresponding audio file and *** are characters that needs to be there but are not important


## Acknowledgment
A huge thank you to Fred Ngole-Mboula for his help and valuable advice.