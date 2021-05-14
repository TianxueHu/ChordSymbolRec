"""
Simple Baseline Model for AV-Sync. Organinzed in PyTorch Lightning
Flatten both audio and video features; concat them and feed into sequential linear layers.
"""
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class LitBaseline(pl.LightningModule):
    def __init__(self, vec_dim, chord_vocab, configs):
        super().__init__()
        
        self.vec_dim = vec_dim
        self.chord_vocab = chord_vocab
        self.vocab_size = len(self.chord_vocab.stoi)

        self._init_configs(configs)
        self._init_model()

        self.save_hyperparameters("vec_dim", "configs")

    def _init_configs(self, configs):
        self.dataset_name = configs.dataset.name
        self.hidden_dim = configs.model.hidden_dim
        self.lr = configs.training.lr
        self.dropout = configs.model.dropout
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
            nn.Linear(self.hidden_dim//2, self.vocab_size))

    def forward(self, data):
        
        out = self.fc(data)
        return out

    def configure_optimizers(self):
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum, weight_decay = 5e-4)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer),
            'monitor': 'val_loss',
        }
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

        preds = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        return self.vec_decode(preds), self.vec_decode(labels)


    def validation_epoch_end(self, validation_step_outputs):
        preds, labels = zip(*validation_step_outputs)
        preds = np.vstack(preds)
        labels = np.vstack(labels)

        ### Get chord name accuracy ###
        chord_name_acc = np.sum(preds == labels) / np.product(labels.shape)

        ### Get root and quality acc ###
        root_preds = preds.copy()
        quality_preds = preds.copy()
        for r_id in range(preds.shape[0]):
            for c_id in range(preds.shape[1]):
                sp = preds[r_id, c_id].split(' ')
                root_preds[r_id, c_id] = sp[0]
                quality_preds[r_id, c_id] = ' '.join(sp[1:])
            
        root_labels = labels.copy()
        quality_labels = labels.copy()
        for r_id in range(labels.shape[0]):
            for c_id in range(labels.shape[1]):
                sp = labels[r_id, c_id].split(' ')
                root_labels[r_id, c_id] = sp[0]
                quality_labels[r_id, c_id] = ' '.join(sp[1:])


        root_acc = np.sum(root_preds == root_labels) / np.product(root_preds.shape)
        quality_acc = np.sum(quality_preds == quality_labels) / np.product(quality_preds.shape)

        self.log("val_name_acc", chord_name_acc, on_epoch = True, prog_bar = True)
        self.log("val_root_acc", root_acc, on_epoch = True, prog_bar = True)
        self.log("val_quality_acc", quality_acc, on_epoch = True, prog_bar = True)

    def test_step(self, batch, batch_idx):
        data, labels = batch
        prob = self(data)
        test_loss = self.criterion(prob, labels) # need to be named loss?
        predictions = torch.argmax(prob, axis = -1)
        self.log("test_loss", test_loss, on_epoch = True, prog_bar = True)
        self.log("test_acc", self.test_acc(predictions, labels), on_epoch = True, prog_bar = True)

        return test_loss

    def decode(self, x):
        return self.chord_vocab.itos[x]

    def vec_decode(self, x):
        return np.vectorize(self.decode)(x)
    