import torch.nn as nn

from .encoder import *
from .decoder import *
from .attention import *
 
class NMTModelBA(nn.Module):
    def __init__(self,
                 attn_concat_dim,
                 attn_latent_dim,
                 vocab_size,
                 latent_dim = 1000,
                 num_enc_layers = 1,
                 dropout = 0,
                 bidirectional = True,
                 padding_idx = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

        self.attn_concat_dim = attn_concat_dim
        self.attn_latent_dim = attn_latent_dim

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.num_enc_layers = num_enc_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx

        self.create_components()

    def create_components(self):
        self.attn_block = BahdanauAttention(self.attn_concat_dim, self.attn_latent_dim)

        self.encoder = NMTEncoder(
            self.vocab_size,
            latent_dim = self.latent_dim,
            num_enc_layers = self.num_enc_layers,
            dropout = self.dropout,
            bidirectional = self.bidirectional,
            padding_idx = self.padding_idx,
            **self.kwargs
        )
        self.decoder = NMTDecoderBA(
            self.vocab_size,
            latent_dim = self.latent_dim,
            num_enc_layers = self.num_enc_layers,
            dropout = self.dropout,
            bidirectional = self.bidirectional,
            padding_idx = self.padding_idx,
            **self.kwargs
        )

        self.linear = nn.Linear(
            in_features = 2 * self.latent_dim if self.bidirectional else self.latent_dim,
            out_features = self.vocab_size + 4
        )

    # def context_concat(self, embeddings, context):

    def forward(self, inp_par_ids, inp_comm_ids):
        encoder_output, (encoder_hidden_state, encoder_cell_state) = self.encoder(inp_par_ids)

        decoder_hidden_state, decoder_cell_state = encoder_hidden_state, encoder_cell_state

        # decoder_outputs, (decoder_hidden_state, decoder_cell_state) = self.decoder(inp_comm_ids, encoder_output, decoder_hidden_state, decoder_cell_state)

        # return decoder_outputs

        decoder_outputs = None
        for i in range(inp_comm_ids.shape[1]):
            context = self.attn_block(encoder_output, decoder_hidden_state)

            decoder_output, (decoder_hidden_state, decoder_cell_state) = self.decoder(inp_comm_ids[..., i:i+1], context, decoder_hidden_state, decoder_cell_state)

            # print("Inside model --> ", decoder_output.shape, decoder_hidden_state.shape, decoder_cell_state.shape)
            output_ids = self.linear(decoder_output)
            # print(output_ids.shape)

            # decoder_outputs.append(output_ids)

            if decoder_outputs is None:
                decoder_outputs = output_ids
            else:
                decoder_outputs = torch.cat( [decoder_outputs, output_ids] , dim = 1)

        # print("decoder_outputs -->", decoder_outputs.shape)
        return decoder_outputs



class NMTModelLA(nn.Module):
    def __init__(self,
                 vocab_size,
                 latent_dim = 1000,
                 num_enc_layers = 1,
                 dropout = 0,
                 bidirectional = True,
                 padding_idx = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.num_enc_layers = num_enc_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx

        self.create_components()

    def create_components(self):
        self.attn_block = LuongAttention()

        self.encoder = NMTEncoder(
            self.vocab_size,
            latent_dim = self.latent_dim,
            num_enc_layers = self.num_enc_layers,
            dropout = self.dropout,
            bidirectional = self.bidirectional,
            padding_idx = self.padding_idx,
            **self.kwargs
        )
        self.decoder = NMTDecoderLA(
            self.vocab_size,
            latent_dim = self.latent_dim,
            num_enc_layers = self.num_enc_layers,
            dropout = self.dropout,
            bidirectional = self.bidirectional,
            padding_idx = self.padding_idx,
            **self.kwargs
        )

        self.linear = nn.Linear(
            in_features = 4 * self.latent_dim if self.bidirectional else 2 * self.latent_dim,
            out_features = self.vocab_size + 4
        )

    # def context_concat(self, embeddings, context):

    def forward(self, inp_par_ids, inp_comm_ids):
        encoder_outputs, (encoder_hidden_state, encoder_cell_state) = self.encoder(inp_par_ids)

        decoder_hidden_state, decoder_cell_state = encoder_hidden_state, encoder_cell_state

        decoder_outputs, (decoder_hidden_state, decoder_cell_state) = self.decoder(inp_comm_ids, decoder_hidden_state, decoder_cell_state)

        context = self.attn_block(encoder_outputs, decoder_outputs)

        hidden_states_tilda = torch.cat([context, decoder_outputs], dim = -1)

        return self.linear(hidden_states_tilda)

        
    





