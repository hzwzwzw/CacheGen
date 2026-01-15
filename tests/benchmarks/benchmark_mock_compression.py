import torch
import time
import os
import numpy as np
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata

def benchmark_compression():
    # 1. Configuration
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    chunk_size = 128
    num_layers = 32
    num_heads = 32
    head_size = 128
    fmt = "vllm"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Benchmarking CacheGen on {device}")
    print(f"Shape: [Layers:{num_layers}, Tokens:{chunk_size}, Heads:{num_heads}, Dim:{head_size}]")

    # Mock Data: [num_layers, 2, num_tokens, num_heads, head_size]
    # Using float16 for realistic simulation
    mock_kv = torch.randn((num_layers, 2, chunk_size, num_heads, head_size), device=device, dtype=torch.float16)
    
    methods = ["ac", "lz4", "bit_packing"]
    
    print("\n" + "="*80)
    print(f"{ 'Method':<15} {'Comp Ratio':<15} {'Enc Time (ms)':<15} {'Dec Time (ms)':<15} {'MSE':<15}")
    print("="*80)

    for method in methods:
        # Set Environment Variable
        os.environ["LMCACHE_COMPRESSION_METHOD"] = method
        if method == "bit_packing":
            os.environ["QUANT_LEVEL"] = "1"
        else:
            os.environ["QUANT_LEVEL"] = "3"
        
        # Check lz4
        if method == "lz4":
            try:
                import lz4
            except ImportError:
                print(f"{method:<15} SKIPPED (lz4 missing)")
                continue

        # Skip AC if not on CUDA
        if method == "ac" and device != "cuda":
             print(f"{method:<15} SKIPPED (requires cuda)")
             continue

        # Metadata & Config
        meta = LMCacheEngineMetadata(model_name, 1, 0, fmt)
        config = LMCacheEngineConfig(chunk_size, device, None, "cachegen", False)

        # Serializer & Deserializer
        try:
            serializer = CacheGenSerializer(config, meta)
            deserializer = CacheGenDeserializer(config, meta)

            # Warmup
            for _ in range(2):
                encoded = serializer.to_bytes(mock_kv)
                _ = deserializer.from_bytes(encoded)
            
            # Benchmark Encoding
            start_enc = time.perf_counter()
            for _ in range(5):
                encoded_bytes = serializer.to_bytes(mock_kv)
            end_enc = time.perf_counter()
            avg_enc_time = (end_enc - start_enc) / 5.0
            
            # Benchmark Decoding
            start_dec = time.perf_counter()
            for _ in range(5):
                decoded_kv = deserializer.from_bytes(encoded_bytes)
            end_dec = time.perf_counter()
            avg_dec_time = (end_dec - start_dec) / 5.0
            
            # Metrics
            original_size_bytes = mock_kv.numel() * mock_kv.element_size()
            compressed_size_bytes = len(encoded_bytes)
            compression_ratio = original_size_bytes / compressed_size_bytes
            
            # Correctness (MSE)
            # Ensure shape matches first
            if decoded_kv.shape != mock_kv.shape:
                mse = float('nan')
                print(f"Shape Mismatch: {decoded_kv.shape} vs {mock_kv.shape}")
            else:
                mse = torch.mean((decoded_kv.float() - mock_kv.float()) ** 2).item()
                
            print(f"{method:<15} {compression_ratio:<15.2f} {avg_enc_time*1000:<15.2f} {avg_dec_time*1000:<15.2f} {mse:<15.6f}")

        except Exception as e:
             print(f"{method:<15} FAILED: {e}")
             import traceback
             traceback.print_exc()

if __name__ == "__main__":
    benchmark_compression()