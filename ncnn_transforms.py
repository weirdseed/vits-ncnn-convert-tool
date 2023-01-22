import torch
from torch.nn import functional as F
import commons

def searchsorted(bin_locations, inputs, eps=1e-6):
    #for ncnn
    bin_locations[..., -1] += eps
    inputs_ = inputs[..., None]
    inputs_ = inputs_.expand_as(bin_locations)
    inputs_ge = inputs_ >= bin_locations
    sum_ge = inputs_ge.sum(-1) - 1
    return sum_ge


def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse = False,
                              left=0., right=1., bottom=0., top=1.,
                              min_bin_width=1e-3,
                              min_bin_height=1e-3,
                              min_derivative=1e-3):

    num_bins = unnormalized_widths.shape[1]
    
    widths = F.softmax(unnormalized_widths, -1)
    
    widths = (1 - min_bin_width * num_bins) * widths
    widths = min_bin_width + widths

    cumwidths = widths.cumsum(-1)
    
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    
    #for ncnn
    cumwidths_l = cumwidths[..., 0].unsqueeze(-1) * 0 + left
    cumwidths_m = cumwidths[..., 1:-1]
    cumwidths_r = cumwidths[..., -1].unsqueeze(-1) * 0 +right
    cumwidths = torch.concat([cumwidths_l, cumwidths_m, cumwidths_r],-1)
    
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    
    soft_unnormalized_derivatives = commons.softplus(unnormalized_derivatives)
    
    derivatives = min_derivative + soft_unnormalized_derivatives
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

    cumheights = heights.cumsum(-1)

    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    
    # for ncnn
    cumheights_l = cumheights[..., 0].unsqueeze(-1) * 0 + bottom
    cumheights_m = cumheights[..., 1:-1]
    cumheights_r = cumheights[..., -1].unsqueeze(-1) * 0 + top
    cumheights = torch.concat([cumheights_l, cumheights_m, cumheights_r], -1)

    heights = cumheights[..., 1:] - cumheights[..., :-1]
    bin_idx = searchsorted(cumheights, inputs)
    bin_idx = bin_idx[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]

    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]
    
    a = (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) + input_heights * (input_delta - input_derivatives)
    
    b = input_heights * input_derivatives - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)

    c = input_delta * (input_cumheights - inputs)
    
    discriminant = b.pow(2) - 4 * a * c
    
    root = (-2 * c) / (b + torch.sqrt(discriminant))
    outputs = root * input_bin_widths + input_cumwidths

    return outputs
    