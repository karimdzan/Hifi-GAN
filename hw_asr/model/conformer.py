import torch
from torch import nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, 
                 chan_in, 
                 chan_out, 
                 kernel_size, 
                 padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, 
                 dim = 512, 
                 max_len = 10000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class RelativeMultiHeadAttention(nn.Module):
    def __init__(
            self,
            dim: int = 512,
            heads: int = 16,
            dropout: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        self.dim = dim
        self.dim_head = int(dim / heads)
        self.num_heads = heads
        self.sqrt_dim = math.sqrt(dim)

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, pos_embedding):
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.dim_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.dim_head)

        score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (score + pos_score) / self.sqrt_dim

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.dim)

        return self.out_proj(context)

    def _relative_shift(self, pos_score):
        batch_size, num_heads, s, t = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, s, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, t + 1, s)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, 
                 dim, 
                 heads, 
                 dropout = 0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = RelativeMultiHeadAttention(dim, heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding)

        return self.dropout(outputs)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = (kernel_size // 2, kernel_size // 2 - (kernel_size + 1) % 2)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.ff1 = FeedForward(dim = dim, 
                               mult = ff_mult, 
                               dropout = ff_dropout)
        self.attn = MultiHeadedSelfAttentionModule(dim = dim, 
                                                   heads = heads, 
                                                   dropout = attn_dropout)
        self.conv = ConformerConvModule(dim = dim, 
                                        expansion_factor = conv_expansion_factor, 
                                        kernel_size = conv_kernel_size, 
                                        dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, 
                               mult = ff_mult, 
                               dropout = ff_dropout)

    def forward(self, x):
        x = self.ff1(x) * 0.5 + x
        x = self.attn(x) + x
        x = self.conv(x) + x
        x = self.ff2(x) * 0.5 + x
        x = self.norm(x)
        return x


class Conv2dSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

        return outputs


class Conformer(nn.Module):
    def __init__(
        self,
        n_class,
        dim,
        num_layers,
        input_dim = 128,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.conv_subsample = Conv2dSubsampling(in_channels=1, out_channels=dim)
        self.input_proj = nn.Sequential(
            nn.Linear(dim * (((input_dim - 1) // 2 - 1) // 2), dim),
            nn.Dropout(p=0.1),
        )
        for _ in range(num_layers):
            self.layers.append(ConformerBlock(
                dim = dim,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size
            ))
        self.fc = nn.Linear(dim, n_class, bias=False)

    def forward(self, spectrogram, **batch):
        out = self.conv_subsample(spectrogram)
        out = self.input_proj(out)
        for block in self.layers:
            out = block(out)
        out = self.fc(out)
        return out
    
    def transform_input_lengths(self, input_lengths):
        return ((input_lengths - 1) // 2 - 1) // 2

    