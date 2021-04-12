import random

import torch
import torch.nn as nn
import torch.optim as optim

class BaseSeq2Seq(nn.Module):
    """ The Sequence to Sequence model. """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.device = device

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        
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
        outputs = torch.full((batch_size, seq_len, self.decoder.output_size), start_idx, dtype = torch.float).to(self.device) # problem???
        # outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size).to(self.device)
        encoder_outputs, hidden = self.encoder(source)

        # first input to the decoder is the <sos> token
        input = target[:,0].unsqueeze(1)
        # input = source[:,0]

        for t in range(1, seq_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:,t,:] = output

            
            # input = output.max(1)[1].unsqueeze(1)
            if teacher_forcing:
                input = target[:,t].unsqueeze(1)
            else:
                input = output.max(1)[1].unsqueeze(1)
        # print(outputs)
        return outputs



        

class AttnSeq2Seq(nn.Module):
    """ The Sequence to Sequence model. """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.device = device

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        
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
        outputs = torch.full((batch_size, seq_len, self.decoder.output_size), start_idx, dtype = torch.float).to(self.device) # problem???
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
