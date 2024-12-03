import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import pandas as pd 

def get_gt_plot_waveform(signal, df):
    gt_plot = np.zeros_like(signal)
    cur_sample = int(df.iloc[0].start_time * 16000)
    for i in df.index:
        utt_len = int(df.iloc[i].utt_time * 16000)
        gt_plot[cur_sample:cur_sample+utt_len] = df.iloc[i].speech
        cur_sample += utt_len
        
    plt.figure(figsize=(15, 5))
    plt.plot(gt_plot)
    plt.plot(signal)

def get_plot_spectro(audio_path, melspec, hop_length):
    waveform, sr = torchaudio.load(audio_path)
    df = pd.read_csv(audio_path.replace('.flac', '.csv'))
    mel = melspec(waveform)

    gt_plot = np.zeros((mel.shape[-1]))

    cur_frame = 0
    for i in df.index:
        utt_len = int(round(df.iloc[i].utt_time / (hop_length/sr)))
        gt_plot[cur_frame:cur_frame+utt_len] = df.iloc[i].speech  * 40
        cur_frame += utt_len
        
    plt.figure(figsize=(15, 5))
    plt.plot(gt_plot, color='red')
    plt.pcolormesh(np.log(mel.numpy()[0]))