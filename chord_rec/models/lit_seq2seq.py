"""
Simple Baseline Model for AV-Sync. Organinzed in PyTorch Lightning
Flatten both audio and video features; concat them and feed into sequential linear layers.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

# from chord_rec.models.seq2seq.Seq2Seq import BaseSeq2Seq, AttnSeq2Seq
# from chord_rec.models.seq2seq.Encoder import BaseEncoder
# from chord_rec.models.seq2seq.Decoder import BaseDecoder, AttnDecoder

class BaseEncoder(pl.LightningModule):
    """ The Encoder module of the Seq2Seq model """
    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, n_layers, dropout = 0.5):
        super().__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.n_layers = n_layers
        
        # # self.embedding = nn.Embedding(input_size, emb_size)
        # self.embedding = nn.Linear(input_size, emb_size)

        self.recurrent = nn.LSTM(emb_size, encoder_hidden_size, n_layers, dropout = dropout, batch_first=True)
        
        self.linear1 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len, input_size)

            Returns:
                output (tensor): the output of the Encoder; later fed into the Decoder.
                hidden (tensor): the weights coming out of the last hidden unit
        """

        output, hidden = None, None
        
        # x = self.embedding(input)
        # x = self.dropout(x)

        x = input
        output, (hidden, cell) = self.recurrent(x)
        # hidden = torch.tanh(self.linear2(self.relu(self.linear1(hidden))))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, (hidden, cell)
class AttnDecoder(pl.LightningModule):
    def __init__(self, emb_size, decoder_hidden_size, output_size, n_layers, max_length, dropout = 0.5):
        super().__init__()

        self.emb_size = emb_size
        self.decoder_hidden_size = decoder_hidden_size

        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.emb_size)
        self.attn = nn.Linear(self.decoder_hidden_size + self.emb_size, self.max_length)
        self.attn_combine = nn.Linear(self.decoder_hidden_size + self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(self.dropout)
        self.recurrent = nn.LSTM(emb_size, decoder_hidden_size, n_layers, dropout = dropout, batch_first=True)
        self.out = nn.Linear(self.decoder_hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded.squeeze(1), hidden[0][0]), -1)), dim=-1) # Use the first layer(direction) in hidden
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)
        output = torch.cat((embedded.squeeze(1), attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(1)

        output = F.relu(output)
        output, hidden = self.recurrent(output, hidden)
        
        output = self.out(output.squeeze(1))
        return output, hidden, attn_weights

class AttnSeq2Seq(pl.LightningModule):
    """ The Sequence to Sequence model. """

    def __init__(self, encoder, decoder):
        super().__init__() 


        self.encoder = encoder
        self.decoder = decoder

        
        assert self.encoder.encoder_hidden_size == self.decoder.decoder_hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"


    def forward(self, source, target, out_seq_len = None, teacher_forcing = True, start_idx = None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len, input_size)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        if out_seq_len is None:
            seq_len = source.shape[1]
        
        if start_idx is None:
            start_idx = 0
        outputs = torch.full((batch_size, seq_len, self.decoder.output_size), start_idx, dtype = torch.float)
        # outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size).to(self.device)
        encoder_outputs, hidden = self.encoder(source)

        # first input to the decoder is the <sos> token
        input = target[:,0].unsqueeze(1)
        # input = source[:,0]

        for t in range(1, seq_len):
            output, hidden, attn= self.decoder(input, hidden, encoder_outputs)
            outputs[:,t,:] = output
            # input = output.max(1)[1].unsqueeze(1)
            if teacher_forcing:
                input = target[:,t].unsqueeze(1)
            else:
                input = output.max(1)[1].unsqueeze(1)
        # print(outputs)
        return outputs



class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class LitSeq2Seq(pl.LightningModule):
    def __init__(self, vec_size, max_len, chord_vocab, configs):
        super().__init__()
        
        self.input_size = vec_size

        # TODO: In the future if the inputs are tokens, need to specify this differently
        self.emb_size = vec_size # For already vectorized input
        
        self.max_len = max_len

        self.chord_vocab = chord_vocab
        self.output_size = len(chord_vocab.stoi)

        self._init_configs(configs)
        self._init_model()
        self.save_hyperparameters("vec_size", "max_len", "configs")

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

        self.attn = configs.model.attn
    
    def _init_model(self):
        
        if self.attn:
            # Use Attention
            self.encoder = BaseEncoder(self.input_size, self.emb_size, self.hidden_size, self.hidden_size, self.n_layers, dropout = self.encoder_dropout)        
            self.decoder = AttnDecoder(self.emb_size, self.hidden_size, self.output_size, self.n_layers, self.max_len , dropout = self.decoder_dropout)
            # self.model = AttnSeq2Seq(self.encoder, self.decoder)
            
            # self.model = AttnSeq2Seq(BaseEncoder(self.input_size, self.emb_size, self.hidden_size, self.hidden_size, self.n_layers, dropout = self.encoder_dropout)        
                                    # ,AttnDecoder(self.emb_size, self.hidden_size, self.output_size, self.n_layers, self.max_len , dropout = self.decoder_dropout)
                                    # )

            # print(self.device)
        
        else:
            # Don't use Attention
            self.encoder = BaseEncoder(self.input_size, self.emb_size, self.hidden_size, self.hidden_size, self.n_layers, dropout = self.encoder_dropout)        
            self.decoder = BaseDecoder(self.emb_size, self.hidden_size, self.output_size, self.n_layers, dropout = self.decoder_dropout)
            # self.model = AttnSeq2Seq(self.encoder, self.decoder)
            pass
    
    def forward(self, source, target, out_seq_len = None, teacher_forcing = True, start_idx = None): 
        # return self.model(note, chord, teacher_forcing = teacher_forcing, start_idx = start_idx)

        batch_size = source.shape[0]
        if out_seq_len is None:
            seq_len = source.shape[1]
        
        if start_idx is None:
            start_idx = 0
            
        outputs = torch.full((batch_size, seq_len, self.decoder.output_size), start_idx, dtype = torch.float).to(self.device)
        # outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size).to(self.device)
        encoder_outputs, hidden = self.encoder(source)

        # first input to the decoder is the <sos> token
        input = target[:,0].unsqueeze(1)
        # input = source[:,0]

        for t in range(1, seq_len):
            output, hidden, attn= self.decoder(input, hidden, encoder_outputs)
            outputs[:,t,:] = output
            # input = output.max(1)[1].unsqueeze(1)
            if teacher_forcing:
                input = target[:,t].unsqueeze(1)
            else:
                input = output.max(1)[1].unsqueeze(1)
        # print(outputs)
        return outputs

    def configure_optimizers(self):

        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        elif self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer),
            'monitor': 'val_loss',
        }
    
    def training_step(self, batch, batch_idx):
        note, chord = batch
        chord = chord.long()
        tf = np.random.random()< self.tf_ratios[self.current_epoch - 1]
        prob = self(note, chord, teacher_forcing = tf, start_idx = self.chord_vocab.stoi["<sos>"])
        prob = prob.permute(0,2,1)
        loss = self.criterion(prob, chord)
        
        self.log("loss", loss, on_epoch = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        note, chord = batch
        chord = chord.long()
        tf = False # Don't use tf for evaluation
        prob = self(note, chord, teacher_forcing = tf, start_idx = self.chord_vocab.stoi["<sos>"])
        prob = prob.permute(0,2,1)
        loss = self.criterion(prob, chord)

        self.log("val_loss", loss, on_epoch = True, prog_bar = True)

        preds = prob.detach().cpu().numpy().argmax(axis = 1)
        labels = chord.detach().cpu().numpy()
        preds[:,0] = np.full(len(preds), self.chord_vocab.stoi["<sos>"])
        return self.vec_decode(preds), self.vec_decode(labels)

    def validation_epoch_end(self, validation_step_outputs):
        preds, labels = zip(*validation_step_outputs)
        preds = np.vstack(preds)
        labels = np.vstack(labels)

        ### Get chord name accuracy ###
        mask = (preds != "<sos>") & (preds != "<eos>") & (preds != "<pad>")
        masked_preds = preds[mask]
        masked_labels = labels[mask]

        chord_name_acc = np.sum(masked_preds == masked_labels) / len(masked_labels)


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

        self.log("val_name_acc", chord_name_acc, on_epoch = True, prog_bar = True)
        self.log("val_root_acc", root_acc, on_epoch = True, prog_bar = True)
        self.log("val_quality_acc", quality_acc, on_epoch = True, prog_bar = True)




    def test_step(self, batch, batch_idx):
        note, chord = batch
        chord = chord.long()
        tf = False # Don't use tf for evaluation
        prob = self(note, chord, teacher_forcing = tf, start_idx = self.chord_vocab.stoi["<sos>"])
        prob  = prob .permute(0,2,1)
        loss = self.criterion(prob, chord)
        predictions = torch.argmax(prob, axis = 1)
        self.log("test_loss", loss, on_epoch = True, prog_bar = True)
        self.log("test_acc", self.test_acc(predictions, labels), on_epoch = True, prog_bar = True)
        return test_loss

    def decode(self, x):
        return self.chord_vocab.itos[x]

    def vec_decode(self, x):
        return np.vectorize(self.decode)(x)
    