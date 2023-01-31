import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d

import commons
from ncnn_transforms import rational_quadratic_spline
from commons import init_weights, get_padding, weight_norm

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))
    self.transpose = Transpose()

  def forward(self, x):
    x = self.transpose(x)
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    x = self.transpose(x)
    return x

class ReduceDims(nn.Module):
  def __init__(self) -> None:
    super().__init__()
  def forward(self, x):
    return x.expand_as(x)

class DDSConv(nn.Module):
  """
  Dialted and Depth-Separable Convolution
  """
  def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
    super().__init__()
    self.channels = channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout

    self.drop = nn.Dropout(p_dropout)
    self.convs_sep = nn.ModuleList()
    self.convs_1x1 = nn.ModuleList()
    self.norms_1 = nn.ModuleList()
    self.norms_2 = nn.ModuleList()
    for i in range(n_layers):
      dilation = kernel_size ** i
      padding = (kernel_size * dilation - dilation) // 2
      self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, 
          groups=channels, dilation=dilation, padding=padding
      ))
      self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
      self.norms_1.append(LayerNorm(channels))
      self.norms_2.append(LayerNorm(channels))
    
    self.reducedims = ReduceDims()

  def forward(self, x, x_mask, g=None):
    if g is not None:
      x = x + g
      
    for i in range(self.n_layers):
      y = self.convs_sep[i](x * x_mask)
      
      y = self.norms_1[i](y)
      
      y = F.gelu(y)
      
      y = self.convs_1x1[i](y)
      
      y = self.norms_2[i](y)
      
      y = F.gelu(y)

      x = x + self.reducedims(y)
    
    return x * x_mask

class RandnLike(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, x):
    rand_m = torch.rand_like(x)
    return rand_m

class Embedding(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, x, weight):
    ret = F.embedding(x, weight)
    return ret
    
class WN(torch.nn.Module):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = nn.ModuleList()
    self.res_skip_layers = nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      self.cond_layer = weight_norm(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
      res_skip_layer = weight_norm(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    output = torch.zeros_like(x)
    x_mask = x_mask.expand_as(x)

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)

      
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        g_l = torch.zeros_like(x_in)

      acts = commons.fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          self.hidden_channels)
      
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)

      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        output = output + res_skip_acts

    return output * x_mask

class ResidualReverse(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, stats, x0, x1, x_mask, reverse=False):
    if not reverse:
      logs = torch.zeros_like(stats)
      x1 = stats + x1 * torch.exp(logs)
      x1 = x1 * x_mask.expand_as(x1)
      x = torch.cat([x0, x1], 1)
    else:
      x_mask = x_mask.expand_as(stats)
      stats = stats * x_mask
      # CouplingOut
      x1 = x1 - stats
      x1 = x1 * x_mask
      x = torch.cat([x0, x1], 1)
    return x

class ResidualCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = nn.Conv1d(96, 192, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
    self.post = nn.Conv1d(192, 96, 1)
    self.out = ResidualReverse()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0)
    h = h * x_mask.expand_as(h)
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h)
    stats = stats * x_mask.expand_as(stats) # add to param
    out = self.out(stats, x0, x1, x_mask, reverse)
    return out

class SequenceMask(nn.Module):
  def __init__(self) -> None:
    super().__init__()
  def forward(self, x, x_length):
    return commons.sequence_mask(x_length, x.size(2))

class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
      for c1, c2 in zip(self.convs1, self.convs2):
        xt = F.leaky_relu(x, LRELU_SLOPE)
        if x_mask is not None:
            xt = xt * x_mask
        xt = c1(xt)
        xt = F.leaky_relu(xt, LRELU_SLOPE)
        if x_mask is not None:
            xt = xt * x_mask
        xt = c2(xt)
        x = xt + x
      if x_mask is not None:
        x = x * x_mask
      return x

class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
          xt = F.leaky_relu(x, LRELU_SLOPE)
          if x_mask is not None:
              xt = xt * x_mask
          xt = c(xt)
          x = xt + x
        if x_mask is not None:
          x = x * x_mask
        return x

class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x
    
class Flip(nn.Module):
  def __init__(self) -> None:
    super(Flip,self).__init__()

  def forward(self, x: torch.Tensor, *args, reverse=False, **kwargs):
    x = x.flip(1)
    return x

class ElementwiseAffine(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.channels = channels
    self.m = nn.Parameter(torch.zeros(channels,1))
    self.logs = nn.Parameter(torch.zeros(channels,1))

  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = self.m + torch.exp(self.logs) * x
      y = y * x_mask
      logdet = torch.sum(self.logs * x_mask, [1,2])
      return y, logdet
    else:
      m = self.m.unsqueeze(0).expand_as(x)
      logs = self.logs.unsqueeze(0).expand_as(x)
      x = (x - m) * torch.exp(-logs) * x_mask.expand_as(x)
      return x

class PRQTransform(nn.Module):
  def __init__(self, inverse=True, tails="linear", tail_bound=5.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3, num_bins=10, filter_channels = 192):
    super(PRQTransform, self).__init__()
    self.inverse = inverse
    self.tails = tails
    self.tail_bound = tail_bound
    self.min_bin_width = min_bin_width
    self.min_bin_height = min_bin_height
    self.min_derivative = min_derivative
    self.num_bins = num_bins
    self.filter_channels = filter_channels

  def forward(self, inputs, unnormalized_widths, unnormalized_heights,unnormalized_derivatives):
    constant = math.log(math.exp(1 - 1e-3) - 1)
    
    # for ncnn
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant
    
    o = rational_quadratic_spline(
        inputs=inputs.squeeze(),
        unnormalized_widths=unnormalized_widths.squeeze(),
        unnormalized_heights=unnormalized_heights.squeeze(),
        unnormalized_derivatives=unnormalized_derivatives.squeeze(),
        inverse=self.inverse,
        left=-self.tail_bound, right=self.tail_bound, bottom=-self.tail_bound, top=self.tail_bound,
        min_bin_width=self.min_bin_width,
        min_bin_height=self.min_bin_height,
        min_derivative=self.min_derivative
    )
    
    #for ncnn
    outputs = o.unsqueeze(0).unsqueeze(0)

    return outputs

class Transpose(nn.Module):
  def __init__(self) -> None:
    super().__init__()
  def forward(self, x):
    if len(x.shape) == 3:
      return x.permute(0,2,1)
    return x.T

class ConvFlow(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
    super().__init__()
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.num_bins = num_bins
    self.tail_bound = tail_bound
    self.half_channels = in_channels // 2
    self.transpose = Transpose()
    self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
    self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.)
    self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
    self.transform = PRQTransform(num_bins=num_bins,filter_channels=filter_channels)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)

    h = self.pre(x0) # 1,1,100
    
    h = self.convs(h, x_mask.expand_as(h), g=g)
    h = self.proj(h)
    
    h = h * x_mask.expand_as(h)
    
    h = h.squeeze()
    h = self.transpose(h)
    
    h = h.unsqueeze(0).unsqueeze(0)
    
    unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
     
    unnormalized_heights = h[..., self.num_bins:2*self.num_bins] / math.sqrt(self.filter_channels)

    unnormalized_derivatives = h[..., 2 * self.num_bins:]

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    
    x1 = self.transform(x1, unnormalized_widths, unnormalized_heights,unnormalized_derivatives)
    
    x = torch.cat([x0, x1], 1)

    x_mask = x_mask.expand_as(x)
    x = x * x_mask
    return x
    
    
