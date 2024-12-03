import yaml
import argparse
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch

from lightning.trainer import VAD
from dataset import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

def main(hparams_file):

    # Loading config file from script runs
    cfg = yaml.load(open(args.hparams), Loader=yaml.FullLoader)
    # Make string: ./checkpoins and put name 128_mels because we runs: ./configs/128_mels/
    ckpt_folder = os.path.join('./checkpoints',Path(args.hparams).stem) # ./checkpoints/128_mels 
    # then we make ./checkpoints/128_mels cause there is no such dir
    os.makedirs(ckpt_folder, exist_ok=True)
    # write in ./checkpoints/128_mels/hparams.yml
    # This line need for duplicates to get last meta info
    with open(os.path.join(ckpt_folder, 'hparams.yml'), 'w') as file:
        yaml.dump(cfg, file)
    # Load model
    model = VAD(cfg)
    # Load data 
    datamodule = VADMelDataModule(**cfg['data'])
    # TB Log, to run: tensorboard --logdir=tb_logs/VAD
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="VAD")
    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder, **cfg['model_checkpoint'])
    # Trainer
    trainer = pl.Trainer(**cfg['trainer'],
                         logger=logger,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         enable_checkpointing=True)
    
    # Train
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)