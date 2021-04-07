import random

import torch
import torch.nn as nn
import torch.optim as optim

class BaseEncoder(nn.Module):
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