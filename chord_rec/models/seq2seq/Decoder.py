import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BaseDecoder(nn.Module):
    """ The Decoder module of the Seq2Seq model """
    def __init__(self, emb_size, decoder_hidden_size, output_size, n_layers, dropout = 0.5):
        super().__init__()

        self.emb_size = emb_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, emb_size) # vocab of chord
        self.recurrent = nn.LSTM(emb_size, decoder_hidden_size, n_layers, dropout = dropout, batch_first=True)
            
        self.linear = nn.Linear(decoder_hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden, cell):
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

        output, (hidden, cell) = self.recurrent(x, (hidden, cell))
        output = self.linear(output[:,0])
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden, cell
            

class AttnDecoder(nn.Module):
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
            self.attn(torch.cat((embedded.squeeze(1), hidden[0][0]), -1)), dim=-1)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)
        output = torch.cat((embedded.squeeze(1), attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(1)

        output = F.relu(output)
        output, hidden = self.recurrent(output, hidden)
        
        output = self.out(output.squeeze(1))
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.decoder_hidden_size, device=device)