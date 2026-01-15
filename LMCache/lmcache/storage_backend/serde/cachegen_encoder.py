import abc
import io
import pickle
import torchac
import torchac_cuda
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, List, Any, Optional

from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig, CacheGenEncoderOutput, CacheGenGPUBytestream, CacheGenGPUEncoderOutput, CompressionMethod
import lmcache.storage_backend.serde.cachegen_basics as CGBasics
from lmcache.storage_backend.serde.serde import Serializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

try:
    import lz4.frame
except ImportError:
    lz4 = None

logger = init_logger(__name__)

@_lmcache_nvtx_annotate
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
    xq = torch.round(qA * (C / max1)).to(torch.int8)
    
    x = (xq / C * max1).to(torch.float32)
    
    return xq, max1

@_lmcache_nvtx_annotate
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
    factor = MAX / max1 # shape [nlayers, ntokens, 1]
    xq = torch.round(input_groups * factor + MAX).to(torch.int8) # shape [nlayers, ntokens, nchannels]
    
    return xq, max1

@_lmcache_nvtx_annotate
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

@_lmcache_nvtx_annotate
def encode_ntokens(cdf_int, encode_input, output_buffer, output_lengths) -> torch.Tensor:
    """
    Input:
        cdf_int: int16 tensor on GPU with shape [nlayers, nchannels, Lp]
        encode_input: int8 tensor on GPU with shape [nlayers, ntokens, nchannels]
        output_buffer: uint8 tensor on GPU with shape [nlayers, nchannels, BUFFER_SIZE]
        output_lengths: int32 tensor on GPU with shape [nlayers, nchannels]
    Returns:
        byte_tensor: the byte tensor
    """
    torchac_cuda.encode_fast_new(
            cdf_int,
            encode_input,
            output_buffer,
            output_lengths,
    )
    byte_tensor = collect_bytes(output_buffer, output_lengths)
    return byte_tensor

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

class EncoderBackend(abc.ABC):
    @abc.abstractmethod
    def encode(self, encode_input: torch.Tensor, key_bins: torch.Tensor, value_bins: torch.Tensor, chunk_size: int) -> Tuple[List[CacheGenGPUBytestream], torch.Tensor]:
        """
        Encode the quantized input.
        Returns:
            data_chunks: List of CacheGenGPUBytestream
            cdf: The CDF tensor (if used, else empty)
        """
        pass

class AcEncoderBackend(EncoderBackend):
    def encode(self, encode_input: torch.Tensor, key_bins: torch.Tensor, value_bins: torch.Tensor, chunk_size: int) -> Tuple[List[CacheGenGPUBytestream], torch.Tensor]:
        nlayers = encode_input.shape[0]
        nchannels = encode_input.shape[2]
        
        # NOTE: encode_input MUST be on CUDA for torchac_cuda
        if encode_input.device.type != 'cuda':
             # We might need to error out or move?
             # For now assume the user handles it or we'll get an error from torchac_cuda
             pass

        n_key_layers = nlayers // 2
        new_key = encode_input[:n_key_layers]
        new_value = encode_input[n_key_layers:]
        
        # Calculate CDF (GPU specific via torchac_cuda)
        new_cdf_key = torchac_cuda.calculate_cdf(new_key, int(key_bins.max()))
        new_cdf_value = torchac_cuda.calculate_cdf(new_value, int(value_bins.max()))
        cdf_int = torch.cat([new_cdf_key, new_cdf_value])

        output_buffer = torch.zeros(
                (nlayers, nchannels, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK), 
                dtype=torch.uint8, 
                device=encode_input.device)
        output_lengths = torch.zeros(
                (nlayers, nchannels), 
                dtype=torch.int32, 
                device=encode_input.device)

        data_chunks = []
        for i in range(0, chunk_size, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
            start = i
            end = min(i + CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
            bytestream = encode_ntokens(
                cdf_int,
                encode_input[:, start:end, :],
                output_buffer,
                output_lengths
            )
            data_chunks.append(CacheGenGPUBytestream(
                bytestream = bytestream, 
                bytestream_lengths = output_lengths.clone(),
                ntokens = end - start,
            ))
        return data_chunks, cdf_int

class Lz4EncoderBackend(EncoderBackend):
    def encode(self, encode_input: torch.Tensor, key_bins: torch.Tensor, value_bins: torch.Tensor, chunk_size: int) -> Tuple[List[CacheGenGPUBytestream], torch.Tensor]:
        if lz4 is None:
            raise ImportError("lz4 is not installed. Please install it using `pip install lz4`.")

        nlayers, _, nchannels = encode_input.shape
        data_chunks = []
        
        for i in range(0, chunk_size, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
            start = i
            end = min(i + CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
            
            chunk_data = encode_input[:, start:end, :].cpu().numpy().tobytes()
            compressed = lz4.frame.compress(chunk_data)
            
            bytestream = torch.from_numpy(np.frombuffer(compressed, dtype=np.uint8))
            output_lengths = torch.zeros((nlayers, nchannels), dtype=torch.int32) 
            
            data_chunks.append(CacheGenGPUBytestream(
                bytestream = bytestream,
                bytestream_lengths = output_lengths,
                ntokens = end - start,
            ))
            
        return data_chunks, torch.empty(0)

class BitPackingEncoderBackend(EncoderBackend):
    def encode(self, encode_input: torch.Tensor, key_bins: torch.Tensor, value_bins: torch.Tensor, chunk_size: int) -> Tuple[List[CacheGenGPUBytestream], torch.Tensor]:
        max_bins = max(key_bins.max().item(), value_bins.max().item())
        
        nlayers, _, nchannels = encode_input.shape
        data_chunks = []
        
        for i in range(0, chunk_size, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
            start = i
            end = min(i + CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
            
            chunk_tensor = encode_input[:, start:end, :].cpu()
            
            if max_bins <= 16:
                flat = chunk_tensor.flatten()
                if flat.numel() % 2 != 0:
                    flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8)])
                
                pairs = flat.reshape(-1, 2)
                packed = (pairs[:, 0] << 4) | (pairs[:, 1] & 0x0F)
                bytestream = packed.to(torch.uint8)
                
            else:
                bytestream = chunk_tensor.flatten().to(torch.uint8)

            output_lengths = torch.zeros((nlayers, nchannels), dtype=torch.int32)
            
            data_chunks.append(CacheGenGPUBytestream(
                bytestream = bytestream,
                bytestream_lengths = output_lengths,
                ntokens = end - start,
            ))
            
        return data_chunks, torch.empty(0)

@_lmcache_nvtx_annotate
def encode_function(
        kv: torch.Tensor, 
        config: CacheGenConfig, 
        key_bins: torch.Tensor,
        value_bins: torch.Tensor,
        chunk_size: int) -> CacheGenGPUEncoderOutput:
    """
    Given the path to the original key value cache, encode the KV cache
    """
    num_heads, head_size = kv.shape[-2:]
    fp_k, fp_v = _split_kv(kv)
    nchannels = num_heads * head_size
    nlayers = fp_k.shape[0] + fp_v.shape[0]

    # Quantize
    new_key, max_tensors_key = torch_quant_vectorized(key_bins, fp_k)
    new_value, max_tensors_value = torch_quant_vectorized(value_bins, fp_v)
    encode_input = torch.cat((new_key, new_value), dim=0).reshape(nlayers, chunk_size, nchannels)

    # Select Backend
    method = config.compression_method
    if isinstance(method, str):
        if method == "lz4":
            backend = Lz4EncoderBackend()
        elif method == "bit_packing":
            backend = BitPackingEncoderBackend()
        else:
            backend = AcEncoderBackend()
    else:
         backend = AcEncoderBackend()

    data_chunks, cdf_int = backend.encode(encode_input, key_bins, value_bins, chunk_size)

    return CacheGenGPUEncoderOutput(
            data_chunks,
            cdf_int,
            max_tensors_key = max_tensors_key,
            max_tensors_value = max_tensors_value,
            num_heads = num_heads,
            head_size = head_size,
            compression_method = CompressionMethod(method) if isinstance(method, str) else method
        )

class CacheGenSerializer(Serializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        
        import os
        if "LMCACHE_COMPRESSION_METHOD" in os.environ:
             self.cachegen_config.compression_method = os.environ["LMCACHE_COMPRESSION_METHOD"]

        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.key_third_layers)
        ret.fill_(config.key_third_bins)
        ret[:config.key_second_layers] = config.key_second_bins
        ret[:config.key_first_layers] = config.key_first_bins
        return ret

    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.key_third_layers)
        ret.fill_(config.value_second_bins)
        ret[:config.value_first_layers] = config.value_first_bins
        return ret
        
    @_lmcache_nvtx_annotate
    def to_bytes(
            self,
            tensor: torch.Tensor
        ) -> bytes:
        """
        Serialize a pytorch tensor to bytes.
        """
        if self.fmt == "huggingface":
            tensor = tensor.permute(0, 1, 3, 2, 4)

        ntokens = tensor.shape[2]
        
        key_bins = self.key_bins.to(tensor.device)
        value_bins = self.value_bins.to(tensor.device)
        
        output_dict = encode_function(tensor, self.cachegen_config, 
                                      key_bins, value_bins, ntokens)
        return output_dict.to_bytes()