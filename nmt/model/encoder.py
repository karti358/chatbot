import torch
import torch.nn as nn

class NMTEncoder(nn.Module):
    def __init__(self, vocab_size,
                 latent_dim = 1000,
                 num_enc_layers = 1,
                 dropout = 0,
                 bidirectional = True,
                 padding_idx = None,
                 **kwargs):
        """
        Specify input parameters:
        1. vocab_size
        2. latent_dim, default -> 1000
        3. num_enc_layers, default -> 1
        4. dropout_rate , default -> 0
        5. bidirectional (true/false) , default -> true

        Input:
        inputs -> (batch, timestamps)

        !! Complete execution is batch first, there is no other provision
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.num_enc_layers = num_enc_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings = self.vocab_size + 4,
                                      embedding_dim = self.latent_dim,
                                      padding_idx = self.padding_idx
        )
        self.rnn = nn.LSTM(input_size = self.latent_dim,
                           hidden_size = self.latent_dim,
                           num_layers = self.num_enc_layers,
                           batch_first = True,
                           dropout = self.dropout,
                           bidirectional=self.bidirectional
        )

    # def modify_hidden(self, state):
    #     return torch.unsqueeze( torch.reshape( torch.permute( state, dims = (1, 0, 2) ), shape = (state.shape[1], -1)), dim = 1)
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        # print(embeddings.shape)

        return self.rnn(embeddings)

        # output, (hidden_state, cell_state) = self.rnn(embeddings)
        # return output, ( self.modify_hidden(hidden_state), self.modify_hidden(cell_state) )