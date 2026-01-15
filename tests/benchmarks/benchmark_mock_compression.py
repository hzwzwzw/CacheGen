import torch
import time
import os
import numpy as np
import argparse
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata

def get_real_kv_cache(model_name, prompt, device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers library is required for real model execution. Please install it via `pip install transformers`.")

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    
    print("Generating KV cache...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # run forward pass to get past_key_values
        outputs = model(**inputs, use_cache=True)
    
    past_key_values = outputs.past_key_values
    # past_key_values is ((k_layer_0, v_layer_0), (k_layer_1, v_layer_1), ...)
    # k/v shape: [batch, num_heads, seq_len, head_dim]
    
    # Convert to [num_layers, 2, seq_len, num_heads, head_dim]
    # We assume batch size 1
    
    layers = []
    for k, v in past_key_values:
        # k, v: [1, num_heads, seq_len, head_dim]
        # permute to [seq_len, num_heads, head_dim]
        k = k.squeeze(0).permute(1, 0, 2)
        v = v.squeeze(0).permute(1, 0, 2)
        layers.append(torch.stack([k, v]))
        
    kv_cache = torch.stack(layers) # [num_layers, 2, seq_len, num_heads, head_dim]
    return kv_cache

def benchmark_compression():
    parser = argparse.ArgumentParser(description="Benchmark CacheGen Compression")
    parser.add_argument("--use_real_model", action="store_true", help="Use real model for KV generation")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model name")
    parser.add_argument("--prompt", type=str, default="Hello, how are you? I am doing great today. Tell me a story about a dragon.", help="Prompt text")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()

    # 1. Configuration
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name
    fmt = "vllm"

    print(f"Benchmarking CacheGen on {device}")
    
    if args.use_real_model:
        print(f"Using real model: {model_name} with prompt: '{args.prompt[:50]}...' ")
        kv_cache = get_real_kv_cache(model_name, args.prompt, device)
        num_layers, _, chunk_size, num_heads, head_size = kv_cache.shape
    else:
        print("Using random mock data")
        chunk_size = 128
        num_layers = 32
        num_heads = 32
        head_size = 128
        kv_cache = torch.randn((num_layers, 2, chunk_size, num_heads, head_size), device=device, dtype=torch.float16)

    print(f"Shape: [Layers:{num_layers}, Tokens:{chunk_size}, Heads:{num_heads}, Dim:{head_size}]")

    methods = ["ac", "lz4", "bit_packing"]

    original_size_bytes = kv_cache.numel() * kv_cache.element_size()

    print(f"Origin size: {original_size_bytes / (1024*1024) :.2f} MB")
    
    print("\n" + "="*100)
    print(f"{ 'Method':<15} {'Comp Size (MB)':<15} {'Comp Ratio':<15} {'Enc Time (ms)':<15} {'Dec Time (ms)':<15} {'MSE':<15}")
    print("="*100)
    result = []
    for method in methods:
        # Set Environment Variable
        os.environ["LMCACHE_COMPRESSION_METHOD"] = method
        
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
                encoded = serializer.to_bytes(kv_cache)
                _ = deserializer.from_bytes(encoded)
            
            # Benchmark Encoding
            start_enc = time.perf_counter()
            for _ in range(5):
                encoded_bytes = serializer.to_bytes(kv_cache)
            end_enc = time.perf_counter()
            avg_enc_time = (end_enc - start_enc) / 5.0
            
            # Benchmark Decoding
            start_dec = time.perf_counter()
            for _ in range(5):
                decoded_kv = deserializer.from_bytes(encoded_bytes)
            end_dec = time.perf_counter()
            avg_dec_time = (end_dec - start_dec) / 5.0
            
            # Metrics
            compressed_size_bytes = len(encoded_bytes)
            compression_ratio = original_size_bytes / compressed_size_bytes
            
            # Correctness (MSE)
            # Ensure shape matches first
            if decoded_kv.shape != kv_cache.shape:
                mse = float('nan')
                print(f"Shape Mismatch: {decoded_kv.shape} vs {kv_cache.shape}")
            else:
                mse = torch.mean((decoded_kv.float() - kv_cache.float()) ** 2).item()
                
            print(f"{method:<15} {compressed_size_bytes / (1024*1024) :<15.2f} {compression_ratio:<15.2f} {avg_enc_time*1000:<15.2f} {avg_dec_time*1000:<15.2f} {mse:<15.6f}")
            result.append(decoded_kv)
        except Exception as e:
             print(f"{method:<15} FAILED: {e}")
             import traceback
             traceback.print_exc()
    for res in result:
        for res2 in result:
            print(torch.all(torch.eq(res,res2)))

if __name__ == "__main__":
    benchmark_compression()
