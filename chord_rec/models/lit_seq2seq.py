"""
Simple Baseline Model for AV-Sync. Organinzed in PyTorch Lightning
Flatten both audio and video features; concat them and feed into sequential linear layers.
"""
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from chord_rec.models.seq2seq.Seq2Seq import BaseSeq2Seq, AttnSeq2Seq
from chord_rec.models.seq2seq.Encoder import BaseEncoder
from chord_rec.models.seq2seq.Decoder import BaseDecoder, AttnDecoder


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class LitSeq2Seq(pl.LightningModule):
    def __init__(self, vec_size, vocab_size, max_len, pad_idx, device, configs):
        super().__init__()
        
        self.input_size = vec_size

        # TODO: In the future if the inputs are tokens, need to specify this differently
        self.emb_size = vec_size # For already vectorized input
        self.output_size = vocab_size
        self.max_len = max_len
        self.device = device
        self.pad_idx = pad_idx

        self._init_configs(configs)
        self._init_model()
        self.save_hyperparameters()

    def _init_configs(self, configs):
        ### Configs to add ###
        self.n_layers = configs.model.n_layers
        self.encoder_dropout = configs.model.encoder_dropout
        self.decoder_dropout = configs.model.decoder_dropout

        self.warm_up = configs.training.warm_up
        self.decay_run = configs.training.decay_run
        self.post_run = configs.training.post_run

        self.tf_ratios = np.hstack((np.full(self.warm_up, 1), np.flip(np.linspace(0, 0.75, self.decay_run)), np.zeros(self.post_run)))

        self.dataset_name = configs.dataset.name
        self.backbone = configs.training.backbone
        self.hidden_size = configs.model.hidden_dim
        self.lr = configs.training.lr
        self.momentum = configs.training.momentum
        self.optimizer_type = configs.training.optimizer_type
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
    
    def _init_model(self):
        
        if configs.model.attn:
            # Use Attention
            self.encoder = BaseEncoder(self.input_size, self.emb_size, self.hidden_size, self.hidden_size, self.n_layers, dropout = self.encoder_dropout)        
            self.decoder = AttnDecoder(self.emb_size, self.hidden_size, self.output_size, self.n_layers, self.max_len , dropout = self.decoder_dropout)
            self.model = AttnSeq2Seq(self.encoder, self.decoder, self.device)
        
        else:
            # Don't use Attention
            self.encoder = BaseEncoder(self.input_size, self.emb_size, self.hidden_size, self.hidden_size, self.n_layers, dropout = self.encoder_dropout)        
            self.decoder = BaseDecoder(self.emb_size, self.hidden_size, self.output_size, self.n_layers, dropout = self.decoder_dropout)
            self.model = AttnSeq2Seq(self.encoder, self.decoder, self.device)
            pass
    
    def forward(self, data): 
        return self.model(data)

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
        self._init_model(video_size, audio_size, self.hidden_size)
        self.save_hyperparameters()

    def _init_configs(self, configs):

        self.hidden_size = configs.training.hidden_dim
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

    