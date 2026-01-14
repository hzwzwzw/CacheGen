import io
import pickle
import torchac
import numpy as np
import torch
import concurrent.futures
from typing import Tuple, List, Any

from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig, CacheGenEncoderOutput, CacheGenGPUBytestream, CacheGenGPUEncoderOutput
import lmcache.storage_backend.serde.cachegen_basics as CGBasics
from lmcache.storage_backend.serde.serde import Deserializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger

logger = init_logger(__name__)

def quant(bins: int, xq: torch.Tensor, max1: float):
    C = bins // 2 - 1
    x = (xq / C * max1)#.to(torch.float16)
    return x

def do_dequantize(t: torch.Tensor, bins: torch.Tensor, maxtensors: torch.Tensor):
    """
    t: [nlayers, ntokens, nchannels]
    bins: [nlayers]
    maxtensors: [nlayers, ntokens, 1]
    """
    C = (bins // 2 - 1)[:, None, None]
    t = t.float()
    t = t - C
    t = t / C
    t = t * maxtensors
    return t

def bytes_to_tensor(bs: bytes, device="cpu") -> torch.Tensor:
    np_array = np.frombuffer(bs, dtype=np.uint8)
    concated_string = torch.from_numpy(np_array).to(device)
    return concated_string

def recombine_bytes(bytes_tensor, output_lengths) -> torch.Tensor:
    # Not needed for CPU single-stream decoding but keeping for interface compatibility if needed
    output_buffer_size = CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK
    offsets = output_lengths.flatten().cumsum(0).roll(1).reshape(output_lengths.shape)
    offsets[0][0] = 0
    indexes = torch.arange(output_buffer_size, device=offsets.device).tile((output_lengths.shape[0], output_lengths.shape[1], 1))
    final_indexes = (indexes + offsets[:, :, None]).clamp(max = len(bytes_tensor) - 1)
    return bytes_tensor[final_indexes]


def decode_chunk(
        cdf: torch.Tensor,
        data_chunk: CacheGenGPUBytestream,
        target_buffer: torch.Tensor
    ) -> torch.Tensor:
    """
    Write the decode output in target_buffer
    Expected shape: [nlayers (kv in total), ntokens, nchannels]
    """
    byte_tensor = data_chunk.bytestream
    if isinstance(byte_tensor, torch.Tensor):
        bs = byte_tensor.numpy().tobytes()
    else:
        bs = byte_tensor
        
    ntokens = data_chunk.ntokens
    nlayers, nchannels, Lp = cdf.shape
    
    # Reconstruct the repeated CDF
    cdf_flat = cdf.reshape(-1, Lp)
    cdf_repeated = cdf_flat.repeat_interleave(ntokens, dim=0)
    
    # Decode
    decoded = torchac.decode_int16_normalized_cdf(cdf_repeated, bs)
    
    # decoded is [N] long tensor of symbols
    # Reshape back to [nlayers, nchannels, ntokens] then [nlayers, ntokens, nchannels]
    decoded = decoded.long().reshape(nlayers, nchannels, ntokens)
    decoded = decoded.permute(0, 2, 1)
    
    target_buffer[:] = decoded

def decode_function_gpu(
        cdf: torch.Tensor, 
        data_chunks: List[CacheGenGPUBytestream], 
        layers_in_key: int,
        chunk_size: int, 
        output: torch.Tensor, 
    ):
    # TODO: dtype and shape -- still have 128 and 8
    """
    Given the path to the encoded KV bytestream, decode the KV cache

    Inputs:
        cdf: the cdf tensor, in shape [2 * nlayers, nchannels, bins + 1]
        data_chunks: the data_chunks in the encoder's output
        layers_in_key: number of layers in K (or V) (K/V should have the same number of layers)
        chunk_size: the chunk_size
        output: output buffer, in shape [ntokens, 2 * nlayers * nchannels]

    Outputs:
        key: the decoded key tensor in the shape of (layers, tokens, nchannels)
        value: the decoded value tensor in the shape of (layers, tokens, nchannels)
    """
    nlayers, nchannels, _ = cdf.shape
    output = output.reshape((nlayers, chunk_size, nchannels))

    def process_chunk(data_chunk, start_idx):
        end = start_idx + data_chunk.ntokens
        decode_chunk(cdf, data_chunk, output[:, start_idx:end, :])

    # Parallelize chunk processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        start = 0
        for data_chunk in data_chunks:
            futures.append(executor.submit(process_chunk, data_chunk, start))
            start += data_chunk.ntokens
        
        # Wait for all tasks to complete
        for f in futures:
            f.result()

    out = output.reshape((2, layers_in_key, chunk_size, nchannels))
    key, value = out.float()

    return key, value

class CacheGenDeserializer(Deserializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.output_buffer = None
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


    def get_output_buffer(self, nlayers: int, nchannels: int, ntokens: int):
        if self.output_buffer is None or self.output_buffer.shape[1] != 2 * nlayers * nchannels:
            self.output_buffer = torch.zeros((self.chunk_size, 2 * nlayers * nchannels), dtype=torch.uint8).cpu()
        return self.output_buffer[:ntokens, :]

    def from_bytes(self, bs: bytes) -> torch.Tensor:
        encoder_output = CacheGenGPUEncoderOutput.from_bytes(bs)
        encoder_output.max_tensors_key = encoder_output.max_tensors_key.cpu()
        encoder_output.max_tensors_value = encoder_output.max_tensors_value.cpu()
        encoder_output.cdf = encoder_output.cdf.cpu()

        ntokens = encoder_output.max_tensors_key.shape[1]
        layers_in_key = encoder_output.max_tensors_key.shape[0]
        key, value = decode_function_gpu(
                encoder_output.cdf,
                encoder_output.data_chunks,
                layers_in_key,
                ntokens,
                self.get_output_buffer(encoder_output.cdf.shape[0] // 2, encoder_output.cdf.shape[1], ntokens)
            )

        key = do_dequantize(key, self.key_bins, encoder_output.max_tensors_key)
        value = do_dequantize(value, self.value_bins, encoder_output.max_tensors_value)

        ''' merge key and value back and reshape '''
        nlayers, ntokens, nchannels = key.shape
        
        blob = torch.stack([key, value]) # [2, nlayers, ntokens, nchannels] 
        blob = blob.reshape((2, nlayers, ntokens, encoder_output.num_heads, encoder_output.head_size))\
        
        match self.fmt:
            case "vllm":
                return blob.permute((1, 0, 2, 3, 4)).to(torch.bfloat16) # [nlayers, 2, ntokens, num_heads, head_size]
            case "huggingface":
                return blob.permute((1, 0, 3, 2, 4)).to(torch.float16) # [nlayers, 2, num_heads, ntokens, head_size]
