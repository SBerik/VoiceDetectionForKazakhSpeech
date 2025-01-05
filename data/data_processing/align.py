from energy_vad import get_non_speech
import soundfile as sf 
from tqdm import tqdm
import os
import numpy as np
from IPython.display import clear_output
import time
import librosa
import re
from label import custom_get_df_from_json
import pandas as pd

import random 
random.seed(42)

def __reconstruct_signal(signal, vad, hop_len):
    reconstructed_signal = []
    for i in range(len(vad)):
        reconstructed_signal.append(signal[i*hop_len:(i+1)*hop_len] * vad[i])
    x = np.concatenate(reconstructed_signal)
    x = x[x != 0]
    return x

def custom_create_noise_samples (tracklist, where_to_save, seeed, total_duration=2995125):
    sum_durations = 0
    i = 0 # Это нужно что бы повторные файлы содержали разные имена но начинка была одинакова
    for t in tqdm(tracklist):
        t = t.replace('\\', '/')
        signal, vad = get_non_speech(t)
        noise_signal = __reconstruct_signal(signal, vad, hop_len=int(16000*0.025))
        duration = len(noise_signal)/16000
        if 1.0 < duration < 3.0:
            sf.write(where_to_save + 'noise_{}_id{}_sd{}.flac'.format(os.path.basename(t.replace('.flac', '')), i, seeed), noise_signal, samplerate=16000)
            sum_durations += duration
        if sum_durations > total_duration:
            break
        i += 1
    print(f'Sum durations: {sum_durations}')


def add_noise_to_track(audio_path, json_sets, csv_sets, raw_base_path, path_to_save, coeff=3):
    # get audio
    signal, sr = librosa.load(audio_path, sr=16000)
    # get name 
    name_with_extension = os.path.basename(audio_path)
    # get postfix
    postfix = re.sub(r'.*ISSAI_KSC2', '', audio_path)
    json_file = json_sets + postfix.replace('.flac', '.json')
    # get non-speech times 
    df = custom_get_df_from_json(json_file, raw_base_path)
    # Check for empty non-speech segments
    time_indices = list(df[df.speech == 0].index)
    if not time_indices: 
        return None, None
    time_idx = sorted(random.choices(time_indices, k=coeff))
    # Add noises
    noise_file_idx = 0
    for idx in time_idx:
        noise, sr = librosa.load(noise_files[noise_file_idx], sr=16000)
        noise_len = len(noise) / sr
        noise = noise - np.mean(noise)
        time_sample = int(df.iloc[idx].start_time * sr)
        signal = np.concatenate([signal[:time_sample], noise, signal[time_sample:]])
        noise_file_idx += 1
        # modify df
        df.loc[df.index > idx, 'start_time'] += noise_len
        df.loc[df.index == idx, 'utt_time'] += noise_len
        df.loc[df.index >= idx, 'end_time'] += noise_len

    df['audio_id'] = path_to_save + name_with_extension
    return signal, df


def __isSingleClassSample(df: pd.DataFrame) -> bool:
    return df['speech'].eq(0).any()

def getSamplesWithOneClasses(jsons, raw_base_path):      
    single_samples = []
    for json_name in tqdm(jsons):
        json_name = json_name.replace('\\', '/')
        speech_df = custom_get_df_from_json(json_name, raw_base_path)
        if not __isSingleClassSample(speech_df):
            single_samples.append(json_name)
    return single_samples