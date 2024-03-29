import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self,concat_dim, latent_dim,  **kwargs):
        """
        Input Parameters:
        1. concat_dim : Dimension of inputs
        2. output_dim : DImension of the outputs (or) vocab_size
        3. latent_dim : Intermediate dim for linear layer
        """
        super().__init__(**kwargs)
        self.concat_dim = concat_dim
        # self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.linear_1 = nn.Linear(in_features = self.concat_dim,
                                  out_features = self.latent_dim,
                                  bias = True
        )
        self.activation_1 = nn.Tanh()
        self.linear_2 = nn.Linear(in_features = self.latent_dim,
                                  out_features = 1,
                                  bias = False
        )
        self.activation_2 = nn.Softmax(dim = 1)
    
    def attn_block(self, inputs):
        out_1 = self.activation_1(self.linear_1(inputs))
        out_2 = self.activation_2(self.linear_2(out_1))
        return out_2

    def modify_hidden(self, state):
        return torch.unsqueeze( torch.reshape( torch.permute( state, dims = (1, 0, 2) ), shape = (state.shape[1], -1)), dim = 1)

    def forward(self, encoder_hidden_states, decoder_hidden_state):
        # print("encoder hidden states -->", encoder_hidden_states.shape)

        decoder_hidden_state = self.modify_hidden(decoder_hidden_state)
        decoder_hidden_state = torch.tile(decoder_hidden_state, (1, encoder_hidden_states.shape[1], 1))
        # print("Decoder hidden state tiled -> ", decoder_hidden_state.shape)

        attn_input = torch.cat([decoder_hidden_state, encoder_hidden_states], dim = 2)
        # print("Concatenated attention input -> ", attn_input.shape)


        attn_output = self.attn_block(attn_input)
        # print("Attention Output -> ", attn_output.shape)
        
        # attn_output = torch.tile(attn_output, dims = (1,1, encoder_hidden_states.shape[-1]))
        # print("Attention Scores -> ", attn_output.shape)

        context = torch.sum(encoder_hidden_states * attn_output, dim = (1, ) , keepdim = True)
        # print("Context vectors -> ", context.shape)

        return context

class LuongAttention(nn.Module):
    def __init__(self, **kwargs):
        """
        Input Parameters:
        1. concat_dim : Dimension of inputs
        2. output_dim : DImension of the outputs (or) vocab_size
        3. latent_dim : Intermediate dim for linear layer
        """
        super().__init__(**kwargs)

    def forward(self, encoder_hidden_states, decoder_hidden_states):
        dot_product = torch.matmul(decoder_hidden_states, encoder_hidden_states.permute([0, 2, 1]))
        prod_enc = dot_product[..., None] * encoder_hidden_states[:, None, :, :]
        attn_scores = torch.sum(prod_enc, dim = 2)
        return attn_scores

