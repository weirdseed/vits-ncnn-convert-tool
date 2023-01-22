import math
import torch
from torch import nn
from torch.nn import functional as F
import commons
import modules
import attentions
from torch.nn import Conv1d, ConvTranspose1d

from commons import init_weights, weight_norm

class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(
                2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, z, noise_scale, g=None,  w=None, reverse=True):
        x = self.pre(x)
        
        if g is not None:
            cond_g = self.cond(g)
            x = x + cond_g.expand_as(x)

        x = self.convs(x, x_mask.expand_as(x))
        
        x = self.proj(x)
        x = x * x_mask.expand_as(x)
        z = z * noise_scale.expand_as(z)
        
        flows = list(reversed(self.flows[:6]))
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=reverse)
        
        z0, z1 = torch.split(z, [1, 1], 1)
        
        return z0

class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.transepose = modules.Transpose()
        self.sequence_mask = modules.SequenceMask()
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.transepose(x)
        x_mask = self.sequence_mask(x, x_lengths).unsqueeze(1)
        x = x * x_mask.expand_as(x)
        x = self.encoder(x, x_mask)
        stats = self.proj(x)
        stats = stats * x_mask.expand_as(stats)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.reverse = False
        self.gin_channels = gin_channels
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels,
                              kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if self.reverse: reverse = True
        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
      x = self.conv_pre(x)
      
      if g is not None:
          cond_g = self.cond(g)
          cond_g = cond_g.expand_as(x)
          x = x + cond_g
      
      for i in range(self.num_upsamples):
          x = F.leaky_relu(x, 0.1)
          # weight_norm(self.ups[i])
          x = self.ups[i](x)
          
          xs = None
          for j in range(self.num_kernels):
              if xs is None:
                  xs = self.resblocks[i*self.num_kernels+j](x)
              else:
                  xs += self.resblocks[i*self.num_kernels+j](x)
          x = xs / self.num_kernels
          
      x = F.leaky_relu(x)
      x = self.conv_post(x)
      x = torch.tanh(x)

      return x

class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.sequence_mask = modules.SequenceMask()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size,
                              dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.randn_like = modules.RandnLike()

    def forward(self, x, x_lengths, g=None):
        x_mask = self.sequence_mask(x, x_lengths).unsqueeze(1)
        x = self.pre(x)
        x = x * x_mask.expand_as(x)
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x)
        stats = stats  * x_mask.expand_as(stats)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        randn_m = self.randn_like(m)
        z = (m + randn_m * torch.exp(logs))
        z = z * x_mask.expand_as(z)
        return z, x_mask

class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 **kwargs):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels,
                                 filter_channels, n_heads, n_layers, kernel_size, p_dropout) # 文本编码器

        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                             upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels) # HiFi-GAN V1解码器
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
        self.dp = StochasticDurationPredictor(
            hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels) # 随机时长预测器
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels) # 嵌入层

    def voice_conversion(self, x, raw_sid, tgt_sid):
        g_src = self.emb_g([raw_sid]).unsqueeze(-1)
        g_tgt = self.emb_g([tgt_sid]).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(x, torch.LongTensor([x.size(1)]), g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat[0, 0]

    def forward(self, x, sid=0, noise_scale=.667, noise_scale_w=0.8, length_scale=1):
        x, m_p, logs_p, x_mask = self.enc_p(x, torch.LongTensor([x.size(1)]))
        if self.n_speakers > 0:
            g = self.emb_g(torch.LongTensor([sid])).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
    
        z = torch.randn(x.size(0), 2, x.size(2))

        noise_scale_w = torch.ones_like(z) * noise_scale_w
        
        logw = self.dp(x, x_mask, z, noise_scale=noise_scale_w, g=g,  reverse=True)
        
        w = torch.exp(logw) * x_mask * length_scale

        w_ceil = torch.ceil(w)

        summed = torch.sum(w_ceil, [1, 2])

        mask = summed < 1
        summed[mask] = 1

        y_lengths = summed.long()

        y_mask = commons.sequence_mask(y_lengths, None)
        y_mask = y_mask.unsqueeze(1).to(x_mask.dtype)

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        m_p_rand = torch.randn_like(m_p)

        z_p = m_p + m_p_rand * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :None], g=g)
        return o[0, 0]
