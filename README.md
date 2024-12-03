# Voice Detector for Kazakh Speech

<p align="center">
  <img src="pics/result.png" alt="Results of model output" width="70%" />
</p>

## Overview
<p align="center">
  <img src="pics/vad-fig2.png" alt="Results of model output">
</p>

## Metrics
| Accuracy | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
|  0.9550  |  0.9615   | 0.9444 |  0.9529  | 

## Setup
Thing's need to change before run the command below
- You got annotation .csv files for each audio 
- Audio files in any format that can be processed by `librosa` library.
- Audio file names must match with annotation .csv files.

Before run train and inference, setup vertiual env.
```
mkdir VoiceDetectionForKazakhSpeech
python3 -m venv .venv 
```
For Linux (Ubuntu):
```
source .venv/bin/activate
```
For Window: 
```
.\.venv\Scripts\activate
```
Review requirements.txt and then install it:
```
pip install -r requirements.txt
```
To run `train.py`. Before running: review the config.yml files
```
python train.py -p ./configs/NAME_OF_YOUR_CONFIG_FILE.yml
```
Warining: Be careful where you run the script

To run `inference.py`
```
python inference.py PATH_TO_SAMPLE -plot -s -t THRESHOLD -c checkpoints/NAME_OF_YOUR_CONFIG_FILE -w weights/WEIGHT_FILE_NAME
```
Warning: run the file `inference.py` with the trained model weights and the config file. 

Used GPU: `NVIDIA GeForce RTX 4090`  
Number of GPUs: `1`  
CUDA version: `cu118`

Scheme of data folder:
```
data/
    annotation/ 
        Csv/
            Dev/
            Test/
            Train/
        Json/
            Dev/
            Test/
            Train/
    KSC2/
        Dev/
        Test/
        Train/
    samples/
        5ed8a1c0f3ea2.flac
        5ed8a1c0f3ea2.csv
```