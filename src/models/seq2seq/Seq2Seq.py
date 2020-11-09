import random

import torch
import torch.nn as nn
import torch.optim as optim

class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model. """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)


    def forward(self, source, out_seq_len = None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        if out_seq_len is None:
            seq_len = source.shape[1]
        

        outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size).to(self.device)
        encoder_outputs, hidden = self.encoder(source)

        # first input to the decoder is the <sos> token
        input = source[:,0].unsqueeze(1)
        # input = source[:,0]

        for t in range(1, seq_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:,t,:] = output
            input = output.max(1)[1].unsqueeze(1)

        return outputs



        

