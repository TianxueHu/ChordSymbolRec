"""
Simple Baseline Model for AV-Sync. Organinzed in PyTorch Lightning
Flatten both audio and video features; concat them and feed into sequential linear layers.
"""
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class LitBaseline(pl.LightningModule):
    def __init__(self, vec_dim, vocab_size, configs):
        super().__init__()
        
        self.vec_dim = vec_dim
        self.vocab_size = vocab_size

        self._init_configs(configs)
        self._init_model()

        self.save_hyperparameters()

    def _init_configs(self, configs):
        self.dataset_name = configs.dataset.name
        self.hidden_dim = configs.training.hidden_dim
        self.lr = configs.training.lr
        self.dropout = configs.training.dropout
        self.momentum = configs.training.momentum
        self.optimizer_type = configs.training.optimizer_type
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
    
    def _init_model(self):
        
        self.fc = nn.Sequential(
            nn.Linear(self.vec_dim, 2*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, vocab_size))

    def forward(self, data):
        
        out = self.fc(data)
        return out

    def configure_optimizers(self):
        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "SGD":
            return torch.optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum)
    
    def training_step(self, batch, batch_idx):
        data, labels = batch

        prob = self(data)
        loss = self.criterion(prob, labels) # need to be named loss?
        predictions = torch.argmax(prob, axis = -1)
        self.log("loss", loss, on_epoch = True)
        self.log("train_acc", self.train_acc(predictions, labels), prog_bar = True, on_epoch = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        prob = self(data)
        loss = self.criterion(prob, labels) # need to be named loss?
        predictions = torch.argmax(prob, axis = -1)
        self.log("val_loss", loss, on_epoch = True, prog_bar = True)
        self.log("val_acc", self.valid_acc(predictions, labels), on_epoch = True, prog_bar = True)

    def test_step(self, batch, batch_idx):
        data, labels = batch
        prob = self(data)
        test_loss = self.criterion(prob, labels) # need to be named loss?
        predictions = torch.argmax(prob, axis = -1)
        self.log("test_loss", test_loss, on_epoch = True, prog_bar = True)
        self.log("test_acc", self.test_acc(predictions, labels), on_epoch = True, prog_bar = True)

        return test_loss

    