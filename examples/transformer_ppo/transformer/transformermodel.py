import torch
import math
import torch
import torch.nn as nn
from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayerResidual

def make_mlp_default(dim_list, final_nonlinearity=True, nonlinearity="relu"):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if nonlinearity == "relu":
            layers.append(nn.ReLU())
        elif nonlinearity == "tanh":
            layers.append(nn.Tanh())

    if not final_nonlinearity:
        layers.pop()
    return nn.Sequential(*layers)
    
class TransformerModel(nn.Module):
    # feature_size: dimension of input feature size
    # output_size: action dimension (normally 1)
    # ninp: dimension of Projection
    # nhead: number of attention head
    # nhid: hidden layer size of transformer encoder
    # nlayers: number of hidden layers of transformer encoder
    # condition_decoder: use input as query to inform the output
    # transformer_norm: layer_normalization
    # sequence_size: Robot size: e.g. 5x5
    # other_feature_size : other_feature_size
    def __init__(
        self,
        feature_size,
        output_size,
        sequence_size,
        other_feature_size,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.5,
        args=None,
        use_transformer=None,
        is_actor = True
    ):
        """This model is built upon https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
        super(TransformerModel, self).__init__()
        
        self.args=args
        self.model_type = "Transformer"
        self.seq_len = sequence_size
        self.decoder_input_dim = ninp 
        self.ninp = ninp
        self.use_transformer = True if use_transformer == 'transformer' else False
        self.is_actor = is_actor

        if self.use_transformer:
            # Position embedding
            if self.args.POS_EMBEDDING == "learnt":
                self.pos_embedding = PositionalEncoding(ninp, self.seq_len)
            elif self.args.POS_EMBEDDING == "abs":
                self.pos_embedding = PositionalEncoding1D(ninp, self.seq_len)
            elif self.args.POS_EMBEDDING == "None":
                pass
        
            # Transformer Encoder
            encoder_layers = TransformerEncoderLayerResidual(ninp, nhead, nhid, dropout)

            self.transformer_encoder = TransformerEncoder(
                encoder_layers,
                nlayers,
                norm=nn.LayerNorm(ninp) if self.args.transformer_norm else None,
            )

        # Linear Projection for input features
        self.encoder = nn.Linear(feature_size, self.ninp)
        self.condition_decoder = self.args.condition_decoder
        
        # decoder
        if self.args.use_other_obs_encoder:
            hidden_dim = [32, other_feature_size]
            # Task-related observation encoder  
            self.other_info_encoder = MLPObsEncoder(other_feature_size, hidden_dim)
            self.decoder_input_dim += self.other_info_encoder.obs_feat_dim
        else:
            self.decoder_input_dim += other_feature_size

        if self.condition_decoder:
            self.decoder_input_dim += feature_size

        if not self.is_actor:
            self.decoder = make_mlp_default([self.decoder_input_dim] + [64] + [output_size],
                final_nonlinearity=True, nonlinearity='relu')
        else:
            self.decoder = make_mlp_default([self.decoder_input_dim] + [64] + [output_size],
                final_nonlinearity=True, nonlinearity='tanh')

        self.init_weights()
    
    def reset_seq_size(self, seq_size):
        self.seq_len = seq_size
        if self.use_transformer:
            # Position embedding
            if self.args.POS_EMBEDDING == "learnt":
                self.pos_embedding = PositionalEncoding(self.ninp, self.seq_len)
            elif self.args.POS_EMBEDDING == "abs":
                self.pos_embedding = PositionalEncoding1D(self.ninp, self.seq_len)
            elif self.args.POS_EMBEDDING == "None":
                pass
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder[-2].bias.data.zero_()
        self.decoder[-2].weight.data.uniform_(-initrange, initrange)

    def forward(self, modular_state, other_state,obs_mask,return_attention=False):
        
        # Linear Porjection of local observation

        # (num_modular, batch_size, feature_size)
        obs_embed = self.encoder(modular_state) * math.sqrt(self.ninp)

        _, batch_size, _ = obs_embed.shape
        
        # Linear porjection of other observation

        if self.args.use_other_obs_encoder:
            # (batch_size, embed_size)
            other_obs_embed = self.other_info_encoder(other_state)
        else:
            # (batch_size, embed_size)
            other_obs_embed = other_state

        # (batch_size, embed_size x self.seq_len/num_modular)
        other_obs_embed = other_obs_embed.repeat(self.seq_len, 1)
        # (self.seq_len/num_modular, batch_size, other_feature_size)
        other_obs_embed = other_obs_embed.reshape(self.seq_len, batch_size, -1)

        attention_maps = None

        if self.use_transformer:
            if self.args.POS_EMBEDDING in ["learnt", "abs"]:
                obs_embed = self.pos_embedding(obs_embed)

            if return_attention:
                obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                    obs_embed, src_key_padding_mask=obs_mask)
            else:
                obs_embed_t = self.transformer_encoder(
                    obs_embed, src_key_padding_mask=obs_mask
                )
        
            decoder_input = obs_embed_t
        else:
            decoder_input = obs_embed

        if self.condition_decoder:
            decoder_input = torch.cat([decoder_input, other_obs_embed, modular_state], axis=2)
        else:
            decoder_input = torch.cat([decoder_input, other_obs_embed], axis=2)

        output = self.decoder(decoder_input)
        output = output.permute(1, 0, 2)
        output = output.reshape(batch_size, -1)
        return output, attention_maps

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)

class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)

class MLPObsEncoder(nn.Module):
    """Encoder for other env obs."""

    def __init__(self, obs_dim,hidden_size):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + hidden_size
        self.encoder = make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)
