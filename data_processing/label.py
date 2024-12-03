import librosa
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json
import pandas as pd
import os
from tqdm import tqdm
import re

from energy_vad import read_wav, enframe, nrg_vad, deframe
 
def getTimeFrame (fs, d): 
    return np.linspace(0, len(d) / fs, num=len(d))

def getLabel(d, time_frame, reversed_labels = False):
    '''
    d: 1 -- speech, 0 -- non-speech
    '''
    d = np.where(reversed_labels, np.where(d == 1.0, 0.0, 1.0), d)

    # If you need to round uncomment this line of code
    # time_frame = [round(t_f, 5) for t_f in time_frame]
    
    labels = [{'speech': int(d[0].item()), 'start_time': time_frame[0].item(), 'end_time': time_frame[0].item()}]
    
    if len(d) != len(time_frame):
        min_length = min(len(d), len(time_frame))
        time_frame = time_frame[:min_length]
        d = d[:min_length]

    running_state = int(d[0].item())  

    left = right = 0
    for d_i, time_frame_i in zip(d, time_frame):
        d_i = int(d_i.item())  
        if d_i == running_state:
            labels[-1]['end_time'] = time_frame_i.item()
        else:
            labels.append({'speech': d_i, 'start_time': time_frame_i.item(), 'end_time': time_frame_i.item()})
            running_state = d_i 
        
    return labels

def __savingLableInJson(audio_path, labeled_audio, base_path = 'C:/Users/b.smadiarov/Diploma/VD-KazakhSpeech/data/annotation/'):
    new_json_annotated_f = base_path + re.sub(r'.*ISSAI_KSC2', 'Json', audio_path)
    new_json_annotated_f = new_json_annotated_f.replace('.flac', '.json')
    with open(new_json_annotated_f, 'w') as f:
        json.dump({"speech_segments": labeled_audio}, f, indent=4)

def labelingAudios(in_pth_audio, saving = True, annotation_base_pth = 'C:/Users/b.smadiarov/Diploma/VD-KazakhSpeech/data/annotation/'): 
    tracklist = glob(in_pth_audio)
    for audio_path in tqdm(tracklist):
        audio_path = audio_path.replace('\\', '/') # to change \ to change /
        fs,s = read_wav(audio_path)
        percent_high_nrg = 0.0
        win_len = hop_len = int(fs*0.025)
        sframes = enframe(s,win_len,hop_len)
        vad = nrg_vad(sframes, percent_high_nrg)
        d = deframe(vad, win_len, hop_len)
        common_length = min(s.shape, d.shape)[0] # [0] cause it tuple
        d, s = d[:common_length], s[:common_length]
        time_frame = getTimeFrame(fs, d)
        labeled_audio = getLabel(d, time_frame, reversed_labels = False)
        if saving:
            __savingLableInJson(audio_path, labeled_audio, base_path = annotation_base_pth)  

def custom_get_df_from_json(json_path: str, raw_base_path: str, min_length: float = 0.01) -> pd.DataFrame:
    with open(json_path, 'r') as file:
        data = json.load(file)
    speech_df = pd.DataFrame(data['speech_segments'])
    speech_df['utt_time'] = np.round(speech_df['end_time'] - speech_df['start_time'], 3)
    speech_df['speech'] = speech_df['speech'].astype(int)
    speech_df['start_time'] = np.round(speech_df['start_time'].astype(float), 3)
    speech_df['end_time'] = np.round(speech_df['end_time'].astype(float), 3)
    speech_df = speech_df.sort_values(by='start_time').reset_index(drop=True)
    audio_id = raw_base_path + re.sub(r'.*Json/', '', json_path).replace('.json', '.flac') if raw_base_path else ValueError("Не указан путь до файла!")
    speech_df['audio_id'] = audio_id
    return speech_df

def get_consolidated_dfs(list_jsons, raw_base_path, each_saving = False) -> pd.DataFrame:
    speech_dfs = []
    for audio_path in tqdm(list_jsons):
        audio_path = audio_path.replace('\\', '/') # to change \ to change /
        speech_df = custom_get_df_from_json(audio_path, raw_base_path)
        if each_saving:
            speech_df.to_csv(audio_path.replace('Json', 'Csv').replace('.json', '.csv'), index=False)
        speech_dfs.append(speech_df)
    speech_df = pd.concat(speech_dfs).reset_index(drop=True)
    return speech_df