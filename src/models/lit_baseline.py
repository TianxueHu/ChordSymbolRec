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

    def _sample_data(self, X, y):
        indices = []
        for i in range(ways):
            indices.append(np.random.choice(self.shots*2,self.shots, replace = False) + 2 * i * self.shots)
        mask_indices = np.hstack(indices)   
        mask = np.zeros(X.size(0), dtype=bool)
        mask[mask_indices] = True
        X_oracle = X[mask]
        y_oracle = y[mask]
        X_pseudo = X[~mask]
        y_pseudo = y[~mask]

        return X_oracle, y_oracle, X_pseudo, y_pseudo, mask

class LitBaseline(pl.LightningModule):

    def __init__(self, video_size, audio_size, configs):
        super().__init__()
        self._init_configs(configs)
        self._init_model(video_size, audio_size, self.hidden_dim)
        self.save_hyperparameters()

    def _init_configs(self, configs):

        self.hidden_dim = configs.training.hidden_dim
        self.lr = configs.training.lr
        self.momentum = configs.training.momentum
        self.optimizer_type = configs.training.optimizer_type
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.test_predictions = []
        self.test_shifts = []
        self.test_labels = []

        pass

    def _init_model(self, video_size, audio_size, hidden_size = 128):
        """
            Private function for initializing the model architecture

            Params:
                video_size: iterable
                    the shape of the video representaiton matrix
                audio_size: iterable
                    the shape of the audio representation matrix
                hidden_size: int, optional 

                
        """
        self.video_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.product(video_size), hidden_size)
        )

        self.audio_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.product(audio_size), hidden_size)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size//2, 2))
        self.relu = nn.ReLU()
    

    def forward(self, video, audio):
    
        v_out = self.video_stream(video)
        a_out = self.audio_stream(audio)

        cat_out = torch.cat((v_out, a_out), 1)
        out = self.fc(cat_out)
        
        return out

    def configure_optimizers(self):
        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "SGD":
            return torch.optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum)

    def training_step(self, batch, batch_idx):
        video, audio, labels, shifts = batch
        prob = self(video, audio)
        loss = self.criterion(prob, labels) # need to be named loss?
        predictions = torch.argmax(prob, axis = -1)
        self.log("loss", loss, on_epoch = True)
        self.log("train_acc", self.train_acc(predictions, labels), prog_bar = True, on_epoch = True)
        return loss
    
    # def training_step_end(self, outputs):
    #     self.train_acc(outputs['preds'], outputs['target'])
    #     self.log('train_acc', self.train_acc, on_step=True, prog_bar = True)
    #     self.log("loss", outputs['loss'], on_step=True, prog_bar = True)

    # def training_epoch_end(self, outs):
    #     # log epoch metric
    #     self.log('train_acc_epoch', self.train_acc.compute(), prog_bar = True)

    def validation_step(self, batch, batch_idx):
        video, audio, labels, shifts = batch
        prob = self(video, audio)
        loss = self.criterion(prob, labels) # need to be named loss?
        predictions = torch.argmax(prob, axis = -1)
        self.log("val_loss", loss, on_epoch = True, prog_bar = True)
        self.log("val_acc", self.valid_acc(predictions, labels), on_epoch = True, prog_bar = True)
        # return val_loss, val_acc

    # def validation_epoch_end(self, outs):
    #     self.log('val_acc_epoch', self.valid_acc.compute(), prog_bar = True)

    
    def test_step(self, batch, batch_idx):
        video, audio, labels, shifts = batch
        prob = self(video, audio)
        test_loss = self.criterion(prob, labels) # need to be named loss?
        predictions = torch.argmax(prob, axis = -1)
        self.log("test_loss", test_loss, on_epoch = True, prog_bar = True)
        self.log("test_acc", self.test_acc(predictions, labels), on_epoch = True, prog_bar = True)
        self.test_predictions.append(predictions)
        self.test_shifts.append(shifts)
        self.test_labels.append(labels)

        return test_loss

    