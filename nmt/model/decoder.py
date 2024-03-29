import torch
import torch.nn as nn

class NMTDecoderBA(nn.Module):
    """
    INputs:
    1. 
    """
    def __init__(self,
                 vocab_size,
                 latent_dim = 1000,
                 num_enc_layers = 1,
                 dropout = 0,
                 bidirectional = True,
                 padding_idx = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.num_enc_layers = num_enc_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size + 4,
            embedding_dim = self.latent_dim,
            padding_idx = self.padding_idx
        )

        self.rnn = nn.LSTM(
            input_size = 3 * self.latent_dim if self.bidirectional else 2 * self.latent_dim,
            hidden_size = self.latent_dim,
            num_layers = self.num_enc_layers,
            batch_first = True,
            dropout = self.dropout,
            bidirectional=self.bidirectional
        )

    # def modify_hidden(self, state):
        
    #     return torch.permute( state, dims = (1, 0, 2) )

    
    def forward(self, inputs, context, decoder_hidden_state, decoder_cell_state):
        embeddings = self.embedding(inputs)
        # print("Embeddings -> ", embeddings.shape)
        
        decoder_inputs = torch.cat([embeddings, context], dim = 2)
        # print("Decoder concatenated Inputs -> ", decoder_inputs.shape)

        return self.rnn(decoder_inputs, (decoder_hidden_state, decoder_cell_state))
        
        # outputs , (hidden_state, cell_state) = self.rnn(decoder_inputs, (decoder_hidden_state, decoder_cell_state))

        # print("Outputs shape -> " , outputs.shape)
        # print("Next Hidden State -> ", hidden_state.shape)
        # return outputs, hidden_state , self.modify_hidden(cell_state) )

class NMTDecoderLA(nn.Module):
    """
    INputs:
    1. 
    """
    def __init__(self,
                 vocab_size,
                 latent_dim = 1000,
                 num_enc_layers = 1,
                 dropout = 0,
                 bidirectional = True,
                 padding_idx = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.num_enc_layers = num_enc_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size + 4,
            embedding_dim = self.latent_dim,
            padding_idx = self.padding_idx
        )

        self.rnn = nn.LSTM(
            input_size = self.latent_dim,
            hidden_size = self.latent_dim,
            num_layers = self.num_enc_layers,
            batch_first = True,
            dropout = self.dropout,
            bidirectional=self.bidirectional
        )
    
    def forward(self, inputs, decoder_hidden_state, decoder_cell_state):
        embeddings = self.embedding(inputs)

        return self.rnn(embeddings, (decoder_hidden_state, decoder_cell_state))
