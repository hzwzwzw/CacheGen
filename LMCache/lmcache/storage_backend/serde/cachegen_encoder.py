import io
import pickle
import torchac
import numpy as np
import torch
import concurrent.futures
from dataclasses import dataclass
from typing import Tuple, List, Any

from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig, CacheGenEncoderOutput, CacheGenGPUBytestream, CacheGenGPUEncoderOutput
import lmcache.storage_backend.serde.cachegen_basics as CGBasics
from lmcache.storage_backend.serde.serde import Serializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger

logger = init_logger(__name__)

def torch_quant(bins: int, qA: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Quantize a float tensor to fixed number of bins

    Input:
        bins: number of bins
        qA: the input tensor

    Returns:
        xq: the quantized tensor, in float32
        max1: the maximum value of the tensor
    """
    MAX = bins // 2 - 1
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    max1 = torch.clamp(max1, min=1e-5)
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    
    # x = (xq / C * max1).to(torch.float32)
    
    return xq, max1

def torch_quant_vectorized(bins: torch.Tensor, input_groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize each group of a tensor to fixed number of bins

    Input:
        bins: number of bins for different layers, with shape [nlayer]
        input_groups: with shape [nlayers, ntokens, nchannels]

    Returns:
        quantized groups: [nlayers, ntokens, nchannels]
        maxes: [nlayers, ntokens, 1]
    """
    MAX = (bins // 2 - 1)[:, None, None] # shape [nlayers, 1, 1]
    max1 = torch.amax(torch.abs(input_groups), dim=-1, keepdim=True) # shape [nlayers, ntokens, 1]
    max1 = torch.clamp(max1, min=1e-5)
    factor = MAX / max1 # shape [nlayers, ntokens, 1]
    xq = torch.round(input_groups * factor + MAX).to(torch.int8) # shape [nlayers, ntokens, nchannels]
    
    return xq, max1

def concat_max(max1):
    """
    Given a dict of max tensors, concatenate them into a single tensor
    """
    # TODO: this function can be optimized, we don't really need this
    maxes = []
    for i in range(len(max1)):
        maxes.append(max1[i].unsqueeze(0))
    return torch.cat(maxes, dim=0)

def _split_kv(tensor: torch.Tensor) -> torch.Tensor:
    """
    Split a blob KV tensor to K and V tensors with the merged heads

    Input:
        tensor: the KV tensor with shape [num_layers, 2, num_tokens, num_heads, head_size]

    Returns:
        K and V tensors with shape [num_layers, num_tokens, num_channels]
    """
    num_layers, _, num_tokens, num_heads, head_size = tensor.shape
    return torch.unbind(tensor.reshape(num_layers, 2, num_tokens, num_heads * head_size), dim=1)

def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    """
    Convert floatingpoint CDF to integers. See README for more info.
    """
    PRECISION = 16
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(
      2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
      new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
      r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
      cdf.add_(r)
    return cdf

def calculate_cdf(data, bins):
    # data: [nlayers, ntokens, nchannels], int8
    # bins: int (max bin value)
    nlayers, ntokens, nchannels = data.shape
    data_reshaped = data.permute(0, 2, 1).reshape(-1, ntokens) # [nlayers*nchannels, ntokens]
    
    shift = bins + 1
    
    flat_data = data_reshaped.long()
    row_indices = torch.arange(data_reshaped.shape[0], device=data.device)[:, None]
    flat_indices = flat_data + row_indices * shift
    
    flat_indices = flat_indices.flatten()
    total_bins = data_reshaped.shape[0] * shift
    
    flat_counts = torch.bincount(flat_indices, minlength=total_bins).float()
    counts = flat_counts.reshape(data_reshaped.shape[0], shift)
    
    probs = counts / torch.clamp(counts.sum(dim=1, keepdim=True), min=1.0)
    
    cdf = torch.zeros((data_reshaped.shape[0], bins + 2), dtype=torch.float32, device=data.device)
    cdf[:, 1:] = torch.cumsum(probs, dim=1)
    
    # Normalize to int16
    cdf_int = _convert_to_int_and_normalize(cdf, needs_normalization=True)
    return cdf_int.reshape(nlayers, nchannels, -1)

class CacheGenEncoderImpl:
    def __init__(self, **kwargs) -> None:
        """ 
        Fields: 
        - fp_kv: should be a tensor of shape (num_layers, num_tokens, num_channels)
        - fp_v: should be a tensor of shape (num_layers, num_tokens, num_channels)
        """
        self.fp_k = kwargs["fp_k"].cpu()
        self.fp_v = kwargs["fp_v"].cpu()
        
        self.quantized_key = {}
        self.max_tensors_key = {}  
        self.quantized_value = {}
        self.max_tensors_value = {} 
        self.config = kwargs["config"]
        
    def quantize(self):
        """ Quantize the key and value tensors 
        (self.fp_k and self.fp_v) 
        """
        for layer in range(len(self.fp_k)):
            if layer < self.config["key_first_layers"]:
                bins = self.config["key_first_bins"]
            elif layer < self.config["key_second_layers"]:
                bins = self.config["key_second_bins"]
            else:
                bins = self.config["key_third_bins"]

            tmp = torch_quant(bins, self.fp_k[layer].float())
            self.quantized_key[layer] = tmp[0] + bins // 2 - 1
            self.max_tensors_key[layer] = tmp[1]

        for layer in range(len(self.fp_v)):
            if layer < self.config["value_first_layers"]:
                bins = self.config["value_first_bins"]
            else:
                bins = self.config["value_second_bins"]
            tmp = torch_quant(bins, self.fp_v[layer].float())
            self.quantized_value[layer] = tmp[0]+ bins // 2 - 1
            self.max_tensors_value[layer] = tmp[1]
            
    def compute_cdf(self, is_key):
        """
        Compute the CDF based on the quantized tensors
        Field: 
        - start_layer: the start layer to compute the CDF
        - end_layer: the end layer to compute the CDF
        """
        # TODO: Add start_index here
        channels = self.fp_k[0].shape[-1]
        
        if is_key:
            X = list(self.quantized_key.values())
        else:
            X = list(self.quantized_value.values())

        # Stack layers: [nlayers, ntokens, channels]
        data = torch.stack(X)
        
        value_range = 32
        # Use our new calculate_cdf function
        final_cdf = calculate_cdf(data, value_range)
                
        return final_cdf

def collect_bytes(output_buffer, output_lengths) -> torch.Tensor:
    """
    Collect a byte tensor from the output_buffer + output_lengths
    """
    output_buffer_size = output_buffer.shape[-1]
    flattened_lengths = output_lengths.flatten()
    flattened_buffer = output_buffer.flatten()
    summed_length = (output_buffer_size - flattened_lengths).cumsum(0)
    summed_length = summed_length.roll(1)
    summed_length[0] = 0
    indexes = summed_length.repeat_interleave(flattened_lengths)
    indexes = indexes + torch.arange(len(indexes), device=indexes.device)
    return flattened_buffer[indexes]

def encode_ntokens(cdf_int, encode_input) -> torch.Tensor:
    """
    Input:
        cdf_int: int16 tensor with shape [nlayers, nchannels, Lp]
        encode_input: int8 tensor with shape [nlayers, ntokens, nchannels]
    Returns:
        byte_tensor: the byte tensor
    """
    nlayers, ntokens, nchannels = encode_input.shape
    
    # Permute input to match CDF structure [nlayers, nchannels, ntokens]
    input_perm = encode_input.permute(0, 2, 1).reshape(-1).to(torch.int16) # [nlayers*nchannels*ntokens]
    
    cdf_flat = cdf_int.reshape(-1, cdf_int.shape[-1]) # [nlayers*nchannels, Lp]
    
    # Repeat CDF
    cdf_repeated = cdf_flat.repeat_interleave(ntokens, dim=0) # [N, Lp]
    
    byte_stream = torchac.encode_int16_normalized_cdf(cdf_repeated, input_perm)
    
    np_bytes = np.frombuffer(byte_stream, dtype=np.uint8)
    return torch.from_numpy(np_bytes)

def encode_function(
        kv: torch.Tensor, 
        config: CacheGenConfig, 
        key_bins: torch.Tensor,
        value_bins: torch.Tensor,
        chunk_size: int) -> CacheGenGPUEncoderOutput:
    """
    Given the path to the original key value cache, encode the KV cache
    """
    kv = kv.cpu()
    key_bins = key_bins.cpu()
    value_bins = value_bins.cpu()
    
    num_heads, head_size = kv.shape[-2:]
    fp_k, fp_v = _split_kv(kv)
    nchannels = num_heads * head_size
    nlayers = fp_k.shape[0] + fp_v.shape[0]

    new_key, max_tensors_key = torch_quant_vectorized(key_bins, fp_k)
    new_value, max_tensors_value = torch_quant_vectorized(value_bins, fp_v)
    encode_input = torch.cat((new_key, new_value), dim=0).reshape(nlayers, chunk_size, nchannels)

    cdf_key = calculate_cdf(new_key, int(key_bins.max()))
    cdf_value = calculate_cdf(new_value, int(value_bins.max()))
    cdf_int = torch.cat([cdf_key, cdf_value])

    def process_chunk(i):
        start = i
        end = min(i + CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
        
        chunk_input = encode_input[:, start:end, :]
        bytestream = encode_ntokens(cdf_int, chunk_input)
        
        return CacheGenGPUBytestream(
            bytestream = bytestream, 
            bytestream_lengths = torch.zeros(1), # Dummy
            ntokens = end - start,
        )

    # Parallelize chunk processing
    indices = list(range(0, chunk_size, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data_chunks = list(executor.map(process_chunk, indices))

    return CacheGenGPUEncoderOutput(
            data_chunks,
            cdf_int,
            max_tensors_key = max_tensors_key,
            max_tensors_value = max_tensors_value,
            num_heads = num_heads,
            head_size = head_size,
        )

class CacheGenSerializer(Serializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.key_third_layers)
        ret.fill_(config.key_third_bins)
        ret[:config.key_second_layers] = config.key_second_bins
        ret[:config.key_first_layers] = config.key_first_bins
        return ret.cpu()

    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.key_third_layers)
        ret.fill_(config.value_second_bins)
        ret[:config.value_first_layers] = config.value_first_bins
        return ret.cpu()
        
    def to_bytes(
            self,
            tensor: torch.Tensor
        ) -> bytes:
        """
        Serialize a pytorch tensor to bytes. The serialized bytes should contain
        both the data and the metadata (shape, dtype, etc.) of the tensor.

        Input:
            t: the input pytorch tensor, can be on any device, in any shape,
               with any dtype
        
        Returns:
            bytes: the serialized bytes
        """
        # TODO: permute is expensive here, need a better way to do it at lower level
        if self.fmt == "huggingface":
            tensor = tensor.permute(0, 1, 3, 2, 4)

        ''' expecting a tensor of shape [num_layers, 2, num_tokens, num_heads, head_size] '''
        ntokens = tensor.shape[2]
        output_dict = encode_function(tensor, self.cachegen_config, 
                                      self.key_bins, self.value_bins, ntokens)
        return output_dict.to_bytes()
