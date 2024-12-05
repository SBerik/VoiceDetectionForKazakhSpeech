import torch
import torchmetrics
from trainer import Trainer 
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter as TensorBoard

from utils.load_config import load_config 
from utils.training import metadata_info, configure_optimizer
from models import VADNet 
from dataset import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

def main(hparams_file):
    # Loading config file    
    cfg = load_config(hparams_file)
    # Load data 
    datamodule = VADMelDataModule(**cfg['data']).setup()
    dataloaders = {'train': datamodule.train_dataloader(), 'valid': datamodule.val_dataloader()}
    # Load model
    model = VADNet(**cfg['model'])
    # Meta-data
    metadata_info(model)
    # TensorBoard
    writer = TensorBoard(f'tb_logs/{Path(hparams_file).stem}', comment = f"{cfg['trainer']['ckpt_folder']}")
    # Optimizer
    optimizer = configure_optimizer (cfg, model)
    # Metrics
    metrics = {m: getattr(torchmetrics, m)(task='binary', average='micro').to(cfg['trainer']['device']) for m in ['Accuracy']}
    # Train
    Trainer(**cfg['trainer']).fit(model, dataloaders, torch.nn.BCEWithLogitsLoss(), optimizer, metrics, writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, default="./configs/32_n_frames.yml", help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)