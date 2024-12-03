from glob import glob
import torch as th
import torchaudio
import pytorch_lightning as pl
from typing import Optional
import os
import random
import numpy as np
import math

from utils.get_frame_targets import get_frame_targets

EPS = 1e-8


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
        
        return sample, targets


class VADMelDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=128, train_percent=0.8, valid_percent = 0.1, 
                 test_percent = 0.1, n_frames=256, nfft=400, hop_length=160, n_mels=64, 
                 sr=16000, norm=False, n_workers=4, pin_memory=False, seed = 42, **kwargs):
        super().__init__()
        # Reproducibility
        self._set_seed(seed)
        self.g = th.Generator()
        self.g.manual_seed(seed)
        self.path_list = glob(os.path.join(str(data_dir), '*.flac'))
        random.shuffle(self.path_list)
        assert math.isclose(train_percent + valid_percent + test_percent, 1.0, rel_tol=1e-9), "Sum doesnt equal to 1" 
        self.train_len = int(len(self.path_list) * train_percent)
        self.valid_len = int(len(self.path_list) * valid_percent)
        self.test_len = int(len(self.path_list) * test_percent)
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.nfft = nfft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sr = sr
        self.norm = norm
        self.n_workers = n_workers
        self.pin_memory = pin_memory
    
    def setup(self, stage: Optional[str] = None, info = True):
        # Instantiate sub datasets
        self.train_set = MelVADDataset(self.path_list[:self.train_len], 
                                       n_frames=self.n_frames, 
                                       nfft=self.nfft, 
                                       hop_length=self.hop_length, 
                                       n_mels=self.n_mels, 
                                       sr=self.sr,
                                       norm=self.norm)
        
        self.val_set = MelVADDataset(self.path_list[self.train_len:self.train_len + self.valid_len], 
                                       n_frames=self.n_frames, 
                                       nfft=self.nfft, 
                                       hop_length=self.hop_length, 
                                       n_mels=self.n_mels, 
                                       sr=self.sr,
                                       norm=self.norm)
        
        self.test_set = MelVADDataset(self.path_list[self.train_len + self.valid_len:], 
                                       n_frames=self.n_frames, 
                                       nfft=self.nfft, 
                                       hop_length=self.hop_length, 
                                       n_mels=self.n_mels, 
                                       sr=self.sr,
                                       norm=self.norm)
        
        if info:
            print(f"Size of training set: {len(self.train_set)}")
            print(f"Size of validation set: {len(self.val_set)}")
            print(f"Size of test set: {len(self.test_set)}")
        
        return self # warning
        
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
    
    def test_dataloader(self):
        return th.utils.data.DataLoader(self.test_set,
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