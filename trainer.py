import os

import torch

from utils.measure_time import measure_time
from utils.training import *
from utils.checkpointer import Checkpointer


class Trainer:
    def __init__(self, num_epochs = 100, device='cuda', best_weights = False, checkpointing = False, 
                 checkpoint_interval = 10, model_name = '', path_to_weights= './weights', ckpt_folder = '') -> None:
        os.makedirs(path_to_weights, exist_ok=True)
        self.ckpointer = Checkpointer(model_name, path_to_weights, ckpt_folder)
        self.num_epochs = num_epochs
        self.device = device
        self.best_weights = best_weights
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval          

    @measure_time
    def fit(self, model, dataloaders, criterion, optimizer, metrics, writer) -> None:
        model.to(self.device)
        min_val_loss = float('inf')
        epoch_state = EpochState(metrics)

        for epoch in range(self.num_epochs):
            for phase in ['train', 'valid']:
                
                model.train() if phase == 'train' else model.eval()
                dataloader = dataloaders[phase] 
                running_loss = 0.0
                for m in metrics.keys():
                    metrics[m].reset()
                
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).transpose(1, 2)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    for m in metrics.keys():
                        metrics[m].update(outputs, labels)
                
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_metrics = {m: metrics[m].compute().item() for m in metrics.keys()}
                epoch_state.update_state(epoch_loss, phase, epoch_metrics)
                p_output_log(self.num_epochs, epoch, phase, epoch_state)
                
                if phase == 'valid' and self.best_weights and epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    self.ckpointer.save_best_weight(model, optimizer, epoch, epoch_state)
            
            torch_logger(writer, epoch, epoch_state)
            
            if self.checkpointing and (epoch + 1) % self.checkpoint_interval == 0:
                self.ckpointer.save_checkpoint(model, optimizer, epoch, epoch_state)
