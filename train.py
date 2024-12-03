import torch
import torchmetrics
from trainer import Trainer 
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter as TensorBoard

from utils.load_config import load_config 
from utils.training import metadata_info
from models import VADNet 
from dataset import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(hparams_file):
    # Loading config file    
    cfg, ckpt_folder = load_config(hparams_file)
    # Load data 
    datamodule = VADMelDataModule(**cfg['data']).setup()
    dataloaders = {'train': datamodule.train_dataloader(), 'valid': datamodule.val_dataloader()}
    # Load model
    model = VADNet(**cfg['model'])
    # Meta-data
    metadata_info(model)
    # TensorBoard
    writer = TensorBoard(f'tb_logs/{Path(hparams_file).stem}', comment = f'{ckpt_folder}')
    # Optimizer
    assert cfg['training']["optim"] in ['Adam', 'SGD'], "Invalid optimizer type"
    optimizer = (torch.optim.Adam if cfg['training']["optim"] == 'Adam' else torch.optim.SGD) (model.parameters(), 
                 lr=cfg['training']["lr"], weight_decay=cfg['training']["weight_decay"])
    # Metrics
    metrics = {m: getattr(torchmetrics, m)(task='binary', average='micro').to(device) for m in ['Accuracy']}
    # Train
    Trainer(**cfg['trainer'], ckpt_folder = ckpt_folder).fit(model, dataloaders, torch.nn.BCEWithLogitsLoss(), optimizer, metrics, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # python train.py -p ./configs/32_n_frames.yml
    parser.add_argument("-p", "--hparams", type=str, help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)