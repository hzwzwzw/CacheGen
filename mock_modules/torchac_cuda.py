import torch

def calculate_cdf(tensor, max_val):
    # Return dummy cdf
    # Shape [nlayers, nchannels, max_val+1]
    # Input tensor shape [nlayers, ntokens, nchannels]
    # This mock should output something valid enough for shapes, 
    # but actual values won't be used if we skip AC benchmark.
    nlayers, ntokens, nchannels = tensor.shape
    return torch.zeros((nlayers, nchannels, max_val + 1), dtype=torch.int16, device=tensor.device)

def encode_fast_new(cdf, val, out, out_len):
    pass

def decode_fast_prefsum(cdf, bs, lens, out):
    pass
