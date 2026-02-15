#!/usr/bin/env python3
"""
Benchmark script to find optimal MoE kernel configs for AMD gfx1151.
Generates: E=512,N=512,device_name=AMD-gfx1151,dtype=int4_w4a16.json

Run inside the vLLM toolbox:
  python3 /path/to/tune_moe_gfx1151.py
"""

import json
import time
import itertools
import torch
import triton

# Model params for Qwen3-Next-80B-A3B (E=512, N=512, int4_w4a16)
E = 512  # num experts
N = 512  # intermediate size
K = 512  # hidden size (approximate, will be checked)
DTYPE = "int4_w4a16"
DEVICE = "cuda"

# Batch sizes to tune for (tokens per forward pass)
BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512,
               1024, 1536, 2048, 3072, 4096]

# Search space - conservative for RDNA 4
SEARCH_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128],
    "GROUP_SIZE_M": [1, 4, 8, 16, 32, 64],
    "SPLIT_K": [1],
}

# For larger batch sizes, also test larger M blocks
LARGE_BATCH_EXTRA_M = [128]


def benchmark_config(M, config, num_iters=20, warmup=5):
    """
    Benchmark a single MoE kernel configuration.
    Uses the vLLM fused_moe_kernel directly.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_wna16_triton_kernel,
        moe_align_block_size,
    )

    top_k = 1  # single expert selection for decode
    num_valid = M

    # Create test tensors
    # For WNA16: weights are packed int4 (K x N/2 per expert)
    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    # Packed int4 weights: each byte holds 2 int4 values
    b = torch.randint(0, 255, (E, K, N // 2), dtype=torch.uint8,
                      device=DEVICE)
    # Scales for quantization groups (group_size=128 typical)
    group_size = 128
    num_groups = K // group_size
    b_scales = torch.randn(E, num_groups, N, dtype=torch.float16,
                           device=DEVICE)
    b_zeros = torch.zeros(E, num_groups, N // 2, dtype=torch.uint8,
                          device=DEVICE)

    # Routing: topk expert indices and weights
    topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32,
                             device=DEVICE)
    topk_weights = torch.ones(M, top_k, dtype=torch.float16,
                              device=DEVICE)

    # Output
    c = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)

    # Sorted token indices
    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)

    block_size_m = config["BLOCK_SIZE_M"]

    try:
        # Warmup
        for _ in range(warmup):
            invoke_fused_moe_wna16_triton_kernel(
                A=a,
                B=b,
                C=c,
                B_scale=b_scales,
                B_zp=b_zeros,
                topk_weights=topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                N=N,
                K=K,
                EM=M * top_k,
                num_valid_tokens=num_valid,
                top_k=top_k,
                compute_type=tl_compute_type,
                config=config,
                block_size_m=block_size_m,
                use_moe_wna16_cuda=False,
            )
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            invoke_fused_moe_wna16_triton_kernel(
                A=a,
                B=b,
                C=c,
                B_scale=b_scales,
                B_zp=b_zeros,
                topk_weights=topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                N=N,
                K=K,
                EM=M * top_k,
                num_valid_tokens=num_valid,
                top_k=top_k,
                compute_type=tl_compute_type,
                config=config,
                block_size_m=block_size_m,
                use_moe_wna16_cuda=False,
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters
        return elapsed
    except Exception as e:
        return float('inf')


def simple_benchmark(M, config, num_iters=50, warmup=10):
    """
    Simpler benchmark using matmul as proxy for MoE kernel performance.
    Tests the config parameters against actual triton compilation.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_default_config,
        fused_experts,
    )

    top_k = 1
    try:
        # Create inputs
        hidden = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
        # For the fused experts call, we need properly shaped weights
        w1 = torch.randn(E, N * 2, K, dtype=torch.float16, device=DEVICE)
        w2 = torch.randn(E, K, N, dtype=torch.float16, device=DEVICE)
        topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32,
                                 device=DEVICE)
        topk_weights = torch.ones(M, top_k, dtype=torch.float16,
                                  device=DEVICE)

        torch.cuda.synchronize()

        # Warmup
        for _ in range(warmup):
            fused_experts(hidden, w1, w2, topk_weights, topk_ids,
                          override_config=config)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iters):
            fused_experts(hidden, w1, w2, topk_weights, topk_ids,
                          override_config=config)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters
        return elapsed
    except Exception:
        return float('inf')


def generate_configs(batch_size):
    """Generate config candidates for a given batch size."""
    space = dict(SEARCH_SPACE)
    if batch_size >= 512:
        space["BLOCK_SIZE_M"] = list(
            set(space["BLOCK_SIZE_M"] + LARGE_BATCH_EXTRA_M))

    keys = sorted(space.keys())
    for values in itertools.product(*(space[k] for k in keys)):
        config = dict(zip(keys, values))
        # Skip invalid: BLOCK_SIZE_M should be <= batch_size for efficiency
        # But for decode (M=1), small M is fine
        yield config


def main():
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Tuning MoE config for E={E}, N={N}, dtype={DTYPE}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print()

    best_configs = {}

    for M in BATCH_SIZES:
        print(f"--- Batch size M={M} ---")
        best_time = float('inf')
        best_config = None
        configs_tested = 0

        for config in generate_configs(M):
            configs_tested += 1
            elapsed = simple_benchmark(M, config, num_iters=20, warmup=5)

            if elapsed < best_time:
                best_time = elapsed
                best_config = dict(config)
                print(f"  New best: {elapsed*1000:.3f}ms - {config}")

        if best_config:
            best_configs[str(M)] = best_config
            print(f"  WINNER M={M}: {best_time*1000:.3f}ms - {best_config}")
        else:
            # Use default
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                get_default_config)
            best_configs[str(M)] = get_default_config(
                M, E, N, K, 1, DTYPE)
            print(f"  Using default config for M={M}")
        print()

    # Save config
    output_file = (
        f"E={E},N={N},"
        f"device_name=AMD-gfx1151,"
        f"dtype={DTYPE}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(best_configs, f, indent=4)
    print(f"\nConfig saved to: {output_file}")
    print("Copy this file to the vLLM configs directory:")
    print(f"  /opt/venv/lib/python3.13/site-packages/vllm/"
          f"model_executor/layers/fused_moe/configs/{output_file}")


if __name__ == "__main__":
    import triton.language as tl
    tl_compute_type = tl.float16
    main()
