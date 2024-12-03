import torch as th
import omegaconf as om
import torchmetrics.classification as tm # Note
import pytorch_lightning as pl
from torch import nn

from models import VADNet
from dataset import *

class VAD(pl.LightningModule):
    def __init__(self, hparams: om.DictConfig):
        super().__init__()
        
        self.hparams.update(hparams)
        if not isinstance(hparams, om.DictConfig):
            hparams = om.DictConfig(hparams)
        self.hparams.update(om.OmegaConf.to_container(hparams, resolve=True))
        
        self._device = th.device('cuda' if th.cuda.is_available() else "cpu")
        self.model = VADNet(**self.hparams['model'])
        self.loss = nn.BCEWithLogitsLoss() # default: reduction = "mean"
        self.auroc = tm.BinaryAUROC().to(self._device)
        self.acc = tm.BinaryAccuracy(threshold=0.5).to(self._device)
        self.f1 = tm.BinaryF1Score(threshold=0.5).to(self._device)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim_type = self.hparams.training["optim"]
        assert  optim_type in ['Adam', 'SGD']
        
        if self.hparams.training["optim"] == 'Adam':
            return th.optim.Adam(self.model.parameters() ,lr=self.hparams.training["lr"], weight_decay=self.hparams.training["weight_decay"])
        else: 
            return th.optim.SGD(self.model.parameters() ,lr=self.hparams.training["lr"], weight_decay=self.hparams.training["weight_decay"])

    def training_step(self, batch, batch_idx):
        x, t = batch['spectro'], batch['targets'].squeeze(1)
        probs = self.forward(x).squeeze(-1)
        loss = self.loss(probs, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # Uncomment this when you figured out why you need code squeeze(0) 
        # self.log('train_acc', self.acc(probs, t), on_step=False, on_epoch=True, prog_bar=True)

        # Проверка градиентов
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f'Grad {name}: {param.grad.sum().item()}')
            
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch['spectro'], batch['targets'].squeeze(1)
        probs = self.forward(x).squeeze(-1)
        val_loss = self.loss(probs, t)
    
        probs = probs.squeeze(0)
        t = t.int().squeeze(0)
    
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.acc(probs, t), on_step=False, on_epoch=True, prog_bar=True)
        self.log("auroc", self.auroc(probs, t), on_step=False, on_epoch=True, prog_bar=True)
        self.log("F1", self.f1(probs, t), on_step=False, on_epoch=True, prog_bar=True)
    
        return val_loss