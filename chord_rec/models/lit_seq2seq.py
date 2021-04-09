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
    def __init__(self, vec_size, max_len, chord_vocab, device, configs):
        super().__init__()
        
        self.input_size = vec_size

        # TODO: In the future if the inputs are tokens, need to specify this differently
        self.emb_size = vec_size # For already vectorized input
        
        self.max_len = max_len
        self.device = device
        self.chord_vocab = chord_vocab
        self.output_size = len(chord_vocab.stoi)

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
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.chord_vocab.stoi["<pad>"])
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
    
    def forward(self, note, chord, teacher_forcing, start_idx): 
        return self.model(note, chord, teacher_forcing = teacher_forcing, start_idx = start_idx)

    def configure_optimizers(self):

        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "SGD":
            return torch.optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum)
    
    def training_step(self, batch, batch_idx):
        
        chord = chord.long()
        tf = np.random.random()< self.tf_ratios[self.current_epoch() - 1]
        prob = self(note, chord, teacher_forcing = tf, start_idx = self.chord_vocab.stoi["<sos>"])
        prob  = prob.permute(0,2,1)
        loss = criterion(pred, chord)
        
        self.log("loss", loss, on_epoch = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        chord = chord.long()
        tf = False # Don't use tf for evaluation
        prob = self(note, chord, teacher_forcing = tf, start_idx = self.chord_vocab.stoi["<sos>"])
        prob  = prob .permute(0,2,1)
        loss = criterion(pred, chord)

        self.log("val_loss", loss, on_epoch = True, prog_bar = True)

        preds = prob.detach().cpu().numpy().argmax(axis = -1)
        label = chord.detach().cpu().numpy()
        pred[:,0] = np.full(len(pred), chord_vocab.stoi["<sos>"])

        return self.vec_decode(preds), self.vec_decode(labels)

    def validation_epoch_end(self, validation_step_outputs):
        preds, labels = zip(*validation_step_outputs)
        preds = np.vstack(preds)
        labels = np.vstack(labels)

        ### Get chord name accuracy ###
        mask = (preds != "<sos>") & (preds != "<eos>") & (preds != "<pad>")
        masked_preds = preds[mask]
        masked_labels = labels[mask]

        chord_name_acc = np.sum(masked_preds == masked_labels) / len(masked_labels))


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
        
        mask = (root_preds != "<sos>") & (root_preds != "<eos>") & (root_preds != "<pad>")
        root_preds = root_preds[mask]
        quality_preds = quality_preds[mask]
        root_label = root_labels[mask]
        quality_labels = quality_labels[mask]

        root_acc = np.sum(root_preds == root_label) / len(root_preds)
        quality_acc = np.sum(quality_preds == quality_labels) / len(quality_preds)

        self.log("val_name_acc", loss, on_epoch = True, prog_bar = True)
        self.log("val_root_acc", loss, on_epoch = True)
        self.log("val_quality_acc", loss, on_epoch = True)




    def test_step(self, batch, batch_idx):
        chord = chord.long()
        tf = False # Don't use tf for evaluation
        prob = self(note, chord, teacher_forcing = tf, start_idx = self.chord_vocab.stoi["<sos>"])
        prob  = prob .permute(0,2,1)
        loss = criterion(pred, chord)
        predictions = torch.argmax(prob, axis = -1)
        self.log("test_loss", loss, on_epoch = True, prog_bar = True)
        self.log("test_acc", self.test_acc(predictions, labels), on_epoch = True, prog_bar = True)
        return test_loss

    def decode(self, x):
        return self.chord_vocab.itos[x]

    def vec_decode(self, x):
        return np.vectorize(self.decode)(x)
    