from glob import glob
import torch as th
import pandas as pd
import torchaudio
import pytorch_lightning as pl
from typing import Optional
import os
import random
import numpy as np

import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

EPS = 1e-8

def get_frame_targets(audio_path:str, total_frames:int, hop_length:int, sr:int=16000)->th.Tensor:
    """Aligns groundtruth annotation in seconds to the spectrogram time axis.
       Returns a binary Tensor array of the size of the spectrogram length.

    Args:
        audio_path (str): path to the audio file
        total_frames (int): total frame of the spectrogram
        hop_length (int): hop length parameter for the spectrogram
        sr (int, optional): sample rate. Defaults to 16000.

    Returns:
        th.Tensor: binary Tensor array for groundtruth
    """
    df = pd.read_csv(audio_path.replace('.flac', '.csv'))
    gt = th.zeros(total_frames)

    cur_frame = 0
    for i in df.index:
        utt_len = int(round(df.iloc[i].utt_time / (hop_length / sr)))

        gt[cur_frame:cur_frame + utt_len] = df.iloc[i].speech 
        cur_frame += utt_len

    return gt.unsqueeze(0)


class MelVADDataset(th.utils.data.Dataset):
    def __init__(self, path_list:list, n_frames:int, nfft:int, hop_length:int, n_mels:int, sr:int, norm:bool=False)->th.utils.data.Dataset:

        self.path_list = path_list
        self.sr = sr
        self.mel_spec =  torchaudio.transforms.MelSpectrogram(n_fft=nfft, hop_length=hop_length, n_mels=n_mels)
        self.hop_length = hop_length
        self.n_frames = n_frames
        # TODO
        self.norm = norm

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        # Load track
        track_path = self.path_list[idx]
        audio, _ = torchaudio.load(track_path)
        # LogMelSpec
        spec = th.log(self.mel_spec(audio)+EPS)
        
        # Get spec sample
        # Changed
        offset = 0 if spec.shape[-1] <= self.n_frames else int(th.randint(0, spec.shape[-1] - self.n_frames, [1]))
        
        sample = spec[:, :, offset:offset+self.n_frames]

        # Get targets
        targets = get_frame_targets(track_path, total_frames=spec.shape[-1], hop_length=self.hop_length)
        targets = targets[:, offset:offset+self.n_frames]

        # Delet later
        # Проверяем размеры выборки и меток
        # if sample.shape[-1] != self.n_frames:
        #     print(f"[ERROR] Incorrect sample size: idx={idx}, sample_shape={sample.shape}, expected_frames={self.n_frames}")
        # if targets.shape[-1] != self.n_frames:
        #     print(f"[ERROR] Incorrect targets size: idx={idx}, targets_shape={targets.shape}, expected_frames={self.n_frames}")
        
        if self.norm:
             # TODO
            pass
        else:
            return {"spectro": sample, "targets": targets}


class VADMelDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=128, valid_percent=0.85, n_frames=256, nfft=400, hop_length=160, n_mels=64, sr=16000, norm=False,
                 n_workers=4, pin_memory=False, seed = 42, **kwargs):
        super().__init__()

        # Reproducibility
        self._set_seed(seed)
        self.g = th.Generator()
        self.g.manual_seed(seed)

        # Get train/val split
        self.path_list = glob(os.path.join(str(data_dir), '*.flac'))
        self.split = round(valid_percent*len(self.path_list))
        # self.split_hardcode = 1500
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.nfft = nfft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sr = sr
        self.norm = norm
        self.n_workers = n_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        # Instantiate sub datasets
        self.train_set = MelVADDataset(self.path_list[:self.split], 
                                       n_frames=self.n_frames, 
                                       nfft=self.nfft, 
                                       hop_length=self.hop_length, 
                                       n_mels=self.n_mels, 
                                       sr=self.sr,
                                       norm=self.norm)
        self.val_set = MelVADDataset(self.path_list[self.split:], 
                                       n_frames=self.n_frames, 
                                       nfft=self.nfft, 
                                       hop_length=self.hop_length, 
                                       n_mels=self.n_mels, 
                                       sr=self.sr,
                                       norm=self.norm)

        print(f"Size of training set: {len(self.train_set)}")
        print(f"Size of validation set: {len(self.val_set)}")
        
    def train_dataloader(self):
        return th.utils.data.DataLoader(self.train_set,
                                        batch_size=self.batch_size,
                                        pin_memory=self.pin_memory,
                                        shuffle=True,
                                        num_workers=self.n_workers,
                                        worker_init_fn=self.seed_worker,
                                        generator=self.g)

    def val_dataloader(self):
        return th.utils.data.DataLoader(self.val_set,
                                        batch_size=self.batch_size,
                                        pin_memory=self.pin_memory,
                                        shuffle=False,
                                        num_workers=self.n_workers,
                                        worker_init_fn=self.seed_worker,
                                        generator=self.g)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    def seed_worker(self, worker_id):
        worker_seed = th.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)