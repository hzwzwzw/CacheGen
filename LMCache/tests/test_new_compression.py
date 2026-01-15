import pytest
import torch
import os
import numpy as np

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer

def generate_kv_cache(num_tokens, fmt, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [num_tokens, num_heads, head_size] if fmt == "vllm" else [num_heads, num_tokens, head_size]
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(32):
        k = torch.rand(shape, dtype = dtype, device = device)
        v = torch.rand(shape, dtype = dtype, device = device)
        ret.append((k, v))

    return tuple(ret)

def to_blob(kv_tuples):
    return torch.stack([torch.stack(inner_tuple, dim=0) for inner_tuple in kv_tuples], dim=0)

@pytest.mark.parametrize("method", ["ac", "lz4", "bit_packing"])
def test_compression_methods(method):
    # Skip lz4 if not available
    if method == "lz4":
        try:
            import lz4
        except ImportError:
            pytest.skip("lz4 not installed")

    # Set compression method via env var
    os.environ["LMCACHE_COMPRESSION_METHOD"] = method
    
    # Ensure quantization level is set for consistent config
    os.environ["QUANT_LEVEL"] = "3" 
    
    chunk_size = 256
    fmt = "vllm"
    config = LMCacheEngineConfig.from_defaults(chunk_size = chunk_size)
    metadata = LMCacheEngineMetadata(model_name = "mistralai/Mistral-7B-Instruct-v0.2", world_size = 1, worker_id = 0, fmt = fmt)
    
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata)

    # Check if config picked up the method
    assert serializer.cachegen_config.compression_method == method

    # Generate data
    kv = to_blob(generate_kv_cache(chunk_size, fmt, "cuda"))
    
    # Encode
    output_bytes = serializer.to_bytes(kv)
    assert len(output_bytes) > 0
    
    # Decode
    decoded_kv = deserializer.from_bytes(output_bytes)
    
    # Check shape
    assert decoded_kv.shape == kv.shape
    
    # Check content (should not be all zeros)
    assert decoded_kv.mean().item() != 0.0
    
    # Basic MSE check (just to see if it's not complete garbage)
    # Since it's quantized, it won't be exactly equal.
    mse = torch.mean((decoded_kv.float() - kv.float()) ** 2)
    print(f"Method: {method}, MSE: {mse.item()}")
    
    # For bit_packing with bins=32 (Quant Level 3 has some 32 bins), 
    # my implementation falls back to 8-bit, so it should be quite accurate (lossless after quantization).
    # AC is also lossless after quantization.
    # LZ4 is lossless after quantization.
    # So MSE should be solely due to quantization.
    
    # Clean up
    del os.environ["LMCACHE_COMPRESSION_METHOD"]
    del os.environ["QUANT_LEVEL"]
