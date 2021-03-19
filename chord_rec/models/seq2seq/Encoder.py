import random

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model """
    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout = 0.2, model_type = "RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        
        self.embedding = nn.Embedding(input_size, emb_size)
        
        if model_type == "RNN":
            self.recurrent = nn.RNN(emb_size, encoder_hidden_size, batch_first=True)
        elif model_type == "LSTM":
            self.recurrent = nn.LSTM(emb_size, encoder_hidden_size, batch_first=True)
        
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

        x = self.embedding(input)
        x = self.dropout(x)

        if self.model_type == "RNN":
            output, hidden = self.recurrent(x)
        else:
            output, (hidden, cell) = self.recurrent(x)
        hidden = torch.tanh(self.linear2(self.relu(self.linear1(hidden))))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden