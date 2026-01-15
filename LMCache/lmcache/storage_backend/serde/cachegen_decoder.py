import abc
import io
import pickle
import torchac_cuda
import numpy as np
import torch
from typing import Tuple, List, Any

from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig, CacheGenEncoderOutput, CacheGenGPUBytestream, CacheGenGPUEncoderOutput, CompressionMethod
import lmcache.storage_backend.serde.cachegen_basics as CGBasics
from lmcache.storage_backend.serde.serde import Deserializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.logging import init_logger
import nvtx

try:
    import lz4.frame
except ImportError:
    lz4 = None

logger = init_logger(__name__)

@_lmcache_nvtx_annotate
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
    t = t - C
    t = t / C
    t = t * maxtensors
    return t

@_lmcache_nvtx_annotate
def bytes_to_tensor(bs: bytes, device="cuda") -> torch.Tensor:
    np_array = np.frombuffer(bs, dtype=np.uint8)
    concated_string = torch.from_numpy(np_array).to(device)
    return concated_string

@_lmcache_nvtx_annotate
def recombine_bytes(bytes_tensor, output_lengths) -> torch.Tensor:
    output_buffer_size = CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK
    offsets = output_lengths.flatten().cumsum(0).roll(1).reshape(output_lengths.shape)
    offsets[0][0] = 0
    indexes = torch.arange(output_buffer_size, device=offsets.device).tile((output_lengths.shape[0], output_lengths.shape[1], 1))
    final_indexes = (indexes + offsets[:, :, None]).clamp(max = len(bytes_tensor) - 1)
    return bytes_tensor[final_indexes]


@_lmcache_nvtx_annotate
def decode_chunk_ac(
        cdf: torch.Tensor,
        data_chunk: CacheGenGPUBytestream,
        target_buffer: torch.Tensor
    ) -> torch.Tensor:
    """
    Write the decode output in target_buffer
    Expected shape: [nlayers (kv in total), ntokens, nchannels]
    """
    bytes_tensor = data_chunk.bytestream
    length_prefsum = data_chunk.bytestream_lengths.flatten().cumsum(0).reshape(data_chunk.bytestream_lengths.shape)
    torchac_cuda.decode_fast_prefsum(
            cdf,
            bytes_tensor,
            length_prefsum,
            target_buffer)

class DecoderBackend(abc.ABC):
    @abc.abstractmethod
    def decode(self, cdf: torch.Tensor, data_chunks: List[CacheGenGPUBytestream], output_buffer: torch.Tensor, nlayers: int, nchannels: int, chunk_size: int, key_bins: torch.Tensor, value_bins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the compressed data.
        Returns:
            key: [layers_in_key, chunk_size, nchannels]
            value: [layers_in_value, chunk_size, nchannels]
        """
        pass

class AcDecoderBackend(DecoderBackend):
    def decode(self, cdf: torch.Tensor, data_chunks: List[CacheGenGPUBytestream], output_buffer: torch.Tensor, nlayers: int, nchannels: int, chunk_size: int, key_bins: torch.Tensor, value_bins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # output_buffer is on GPU, shape [chunk_size, 2*nlayers*nchannels]
        
        # Original torchac_cuda decoding
        
        output = output_buffer.reshape((nlayers, chunk_size, nchannels))

        start = 0
        for data_chunk in data_chunks:
            end = start + data_chunk.ntokens
            
            # Note: decode_chunk_ac expects tensors on GPU
            # Ensure everything is on the correct device
            # cdf should be on GPU (set by Deserializer)
            # data_chunk.bytestream should be on GPU
            
            decode_chunk_ac(cdf, data_chunk, output[:, start:end, :])
            start = end

        # The output_buffer is now filled (on GPU).
        
        layers_in_key = nlayers // 2 
        out = output.reshape((2, layers_in_key, chunk_size, nchannels))
        key, value = out.float()
        return key, value

class Lz4DecoderBackend(DecoderBackend):
    def decode(self, cdf: torch.Tensor, data_chunks: List[CacheGenGPUBytestream], output_buffer: torch.Tensor, nlayers: int, nchannels: int, chunk_size: int, key_bins: torch.Tensor, value_bins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if lz4 is None:
            raise ImportError("lz4 is not installed. Please install it using `pip install lz4`.")

        cpu_output = torch.zeros((nlayers, chunk_size, nchannels), dtype=torch.int8)
        
        start = 0
        for data_chunk in data_chunks:
            end = start + data_chunk.ntokens
            
            bytes_data = data_chunk.bytestream.cpu().numpy().tobytes()
            decompressed = lz4.frame.decompress(bytes_data)
            
            chunk_data = np.frombuffer(decompressed, dtype=np.int8)
            chunk_tensor = torch.from_numpy(chunk_data).reshape(nlayers, end-start, nchannels)
            
            cpu_output[:, start:end, :] = chunk_tensor
            start = end
            
        gpu_output = cpu_output.to(output_buffer.device)
        
        layers_in_key = nlayers // 2
        out = gpu_output.reshape((2, layers_in_key, chunk_size, nchannels))
        key, value = out.float()
        return key, value

class BitPackingDecoderBackend(DecoderBackend):
    def decode(self, cdf: torch.Tensor, data_chunks: List[CacheGenGPUBytestream], output_buffer: torch.Tensor, nlayers: int, nchannels: int, chunk_size: int, key_bins: torch.Tensor, value_bins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_bins = max(key_bins.max().item(), value_bins.max().item())
        
        cpu_output = torch.zeros((nlayers, chunk_size, nchannels), dtype=torch.int8)
        
        start = 0
        for data_chunk in data_chunks:
            end = start + data_chunk.ntokens
            chunk_ntokens = end - start
            num_elements = nlayers * chunk_ntokens * nchannels
            
            packed_tensor = data_chunk.bytestream.cpu()
            
            if max_bins <= 16:
                high = (packed_tensor >> 4) & 0x0F
                low = packed_tensor & 0x0F
                
                unpacked = torch.stack([high, low], dim=1).flatten()
                unpacked = unpacked[:num_elements]
                
                unpacked = unpacked.to(torch.int8) 
                
            else:
                unpacked = packed_tensor.to(torch.int8)
            
            chunk_tensor = unpacked.reshape(nlayers, chunk_ntokens, nchannels)
            cpu_output[:, start:end, :] = chunk_tensor
            start = end
            
        gpu_output = cpu_output.to(output_buffer.device)
        layers_in_key = nlayers // 2
        out = gpu_output.reshape((2, layers_in_key, chunk_size, nchannels))
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
        return ret

    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.key_third_layers)
        ret.fill_(config.value_second_bins)
        ret[:config.value_first_layers] = config.value_first_bins
        return ret


    def get_output_buffer(self, nlayers: int, nchannels: int, ntokens: int, device: str = "cuda"):
        if self.output_buffer is None or self.output_buffer.shape[1] != 2 * nlayers * nchannels or self.output_buffer.device.type != device:
            self.output_buffer = torch.zeros((self.chunk_size, 2 * nlayers * nchannels), dtype=torch.uint8).to(device)
        return self.output_buffer[:ntokens, :]

    @_lmcache_nvtx_annotate
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        encoder_output = CacheGenGPUEncoderOutput.from_bytes(bs)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        encoder_output.max_tensors_key = encoder_output.max_tensors_key.to(device)
        encoder_output.max_tensors_value = encoder_output.max_tensors_value.to(device)
        
        self.key_bins = self.key_bins.to(device)
        self.value_bins = self.value_bins.to(device)

        ntokens = encoder_output.max_tensors_key.shape[1]
        layers_in_key = encoder_output.max_tensors_key.shape[0]
        
        # Determine backend
        method = encoder_output.compression_method
        if isinstance(method, str):
            if method == "lz4":
                backend = Lz4DecoderBackend()
            elif method == "bit_packing":
                backend = BitPackingDecoderBackend()
            else:
                backend = AcDecoderBackend()
        else:
             if method == CompressionMethod.LZ4:
                 backend = Lz4DecoderBackend()
             elif method == CompressionMethod.BIT_PACKING:
                 backend = BitPackingDecoderBackend()
             else:
                 backend = AcDecoderBackend()

        if encoder_output.cdf is not None and encoder_output.cdf.numel() > 0:
            nlayers_total = encoder_output.cdf.shape[0]
            nchannels = encoder_output.cdf.shape[1] 
            encoder_output.cdf = encoder_output.cdf.to(device)
        else:
            nlayers_total = layers_in_key * 2
            nchannels = encoder_output.data_chunks[0].bytestream_lengths.shape[1] 

        key, value = backend.decode(
                encoder_output.cdf,
                encoder_output.data_chunks,
                self.get_output_buffer(layers_in_key, nchannels, ntokens, device),
                nlayers_total,
                nchannels,
                ntokens,
                self.key_bins,
                self.value_bins
            )

        key = do_dequantize(key, self.key_bins, encoder_output.max_tensors_key)
        value = do_dequantize(value, self.value_bins, encoder_output.max_tensors_value)

        ''' merge key and value back and reshape '''
        nlayers, ntokens, nchannels = key.shape
        rng = nvtx.start_range("stack KV")
        blob = torch.stack([key, value]) # [2, nlayers, ntokens, nchannels] 
        nvtx.end_range(rng)
        blob = blob.reshape((2, nlayers, ntokens, encoder_output.num_heads, encoder_output.head_size))\
        
        match self.fmt:
            case "vllm":
                return blob.permute((1, 0, 2, 3, 4)).to(torch.bfloat16) # [nlayers, 2, ntokens, num_heads, head_size]
            case "huggingface":
                return blob.permute((1, 0, 3, 2, 4)).to(torch.float16) # [nlayers, 2, num_heads, ntokens, head_size]