import torch
import time
import os
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
    
    # Metadata & Config
    meta = LMCacheEngineMetadata(model_name, 1, 0, fmt)
    config = LMCacheEngineConfig(chunk_size, device, None, "cachegen", False)

    # Serializer & Deserializer
    serializer = CacheGenSerializer(config, meta)
    deserializer = CacheGenDeserializer(config, meta)

    # 2. Benchmark Compression (Encode)
    print("\n--- Benchmarking Compression ---")
    
    # Warmup
    for _ in range(5):
        _ = serializer.to_bytes(mock_kv)
    
    start_time = time.perf_counter()
    iterations = 5
    encoded_bytes = None
    for _ in range(iterations):
        encoded_bytes = serializer.to_bytes(mock_kv)
    end_time = time.perf_counter()
    
    avg_enc_time = (end_time - start_time) / iterations
    original_size_mb = mock_kv.numel() * mock_kv.element_size() / (1024 * 1024)
    compressed_size_mb = len(encoded_bytes) / (1024 * 1024)
    compression_ratio = original_size_mb / compressed_size_mb
    
    print(f"Original Size: {original_size_mb:.2f} MB")
    print(f"Compressed Size: {compressed_size_mb:.2f} MB")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"Avg Encoding Time: {avg_enc_time*1000:.2f} ms")
    print(f"Encoding Throughput: {original_size_mb / avg_enc_time:.2f} MB/s (Original Data)")

    # 3. Benchmark Decompression (Decode)
    print("\n--- Benchmarking Decompression ---")
    
    # Warmup
    for _ in range(5):
        _ = deserializer.from_bytes(encoded_bytes)
        
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = deserializer.from_bytes(encoded_bytes)
    end_time = time.perf_counter()
    
    avg_dec_time = (end_time - start_time) / iterations
    print(f"Avg Decoding Time: {avg_dec_time*1000:.2f} ms")
    print(f"Decoding Throughput: {original_size_mb / avg_dec_time:.2f} MB/s (Original Data)")

    # 4. Correctness Check (Shape)
    decoded_kv = deserializer.from_bytes(encoded_bytes)
    # CacheGenDecoder output might need reshape to match input perfectly or it might come as a Blob
    # Let's check the shape
    print("\n--- Correctness Check ---")
    print(f"Input Shape: {mock_kv.shape}")
    print(f"Output Shape: {decoded_kv.shape}")
    
    # Note: CacheGen uses lossy compression, so exact equality is not expected.
    # We assume 'transmission' is just the bytes moving, which is dominated by bandwidth.
    # The 'Transmission Algorithm' here effectively means 'Compression + Network'.
    # This script isolates the 'Algorithm' part (Compression/Decompression).

if __name__ == "__main__":
    benchmark_compression()
