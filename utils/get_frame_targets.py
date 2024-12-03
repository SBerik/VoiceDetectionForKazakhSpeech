import pandas as pd
import torch

def get_frame_targets(audio_path:str, total_frames:int, hop_length:int, sr:int=16000)->torch.Tensor:
    """Aligns groundtruth annotation in seconds to the spectrogram time axis.
       Returns a binary Tensor array of the size of the spectrogram length.
    Args:
        audio_path (str): path to the audio file
        total_frames (int): total frame of the spectrogram
        hop_length (int): hop length parameter for the spectrogram
        sr (int, optional): sample rate. Defaults to 16000.

    Returns:
        torch.Tensor: binary Tensor array for groundtruth
    """
    df = pd.read_csv(audio_path.replace('.flac', '.csv'))
    gt = torch.zeros(total_frames)

    cur_frame = 0
    for i in df.index:
        utt_len = int(round(df.iloc[i].utt_time / (hop_length / sr)))

        gt[cur_frame:cur_frame + utt_len] = df.iloc[i].speech 
        cur_frame += utt_len

    return gt.unsqueeze(0)