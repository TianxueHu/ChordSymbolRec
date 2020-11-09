import random

import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model """
    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout = 0.2, model_type = "RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        self.embedding = nn.Embedding(output_size, emb_size)

        if model_type == "RNN":
            self.recurrent = nn.RNN(emb_size, decoder_hidden_size, batch_first=True)
        elif model_type == "LSTM":
            self.recurrent = nn.LSTM(emb_size, decoder_hidden_size, batch_first=True)
            
        self.linear = nn.Linear(decoder_hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """

        x = self.embedding(input)
        x = self.dropout(x)
        if self.model_type == "RNN":
            output, hidden = self.recurrent(x, hidden)
        else:
            cell = torch.Tensor(hidden.size())
            output, (hidden, cell) = self.recurrent(x, (hidden, cell))
        output = self.softmax(self.linear(output[:,0]))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
            
