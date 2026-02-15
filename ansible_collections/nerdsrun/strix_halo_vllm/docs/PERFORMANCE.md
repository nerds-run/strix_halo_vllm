# Performance Tuning

Maximize inference speed (tok/s) on AMD Strix Halo gfx1151 APUs.

---

## Hardware Overview

Strix Halo is fundamentally different from discrete GPUs:

| Property | Strix Halo (gfx1151) | Discrete GPU (e.g. MI300X) |
|---|---|---|
| **Architecture** | RDNA 4 iGPU | CDNA 3 |
| **VRAM** | Shared system RAM (up to 128 GB) | Dedicated HBM3e (192 GB) |
| **Bandwidth** | ~200 GB/s (LPDDR5-8000, 8-channel) | ~5.3 TB/s |
| **Compute Units** | 20 SMs | 304 CUs |
| **L2 Cache** | 2 MB | 256 MB |
| **Strength** | Huge unified memory, low cost | Raw throughput |

LLM inference is **memory-bandwidth bound** during token generation. With ~200 GB/s, Strix Halo needs models that minimize the bytes read per token.

---

## Rule #1: Use MoE Models

**Mixture of Experts (MoE)** models activate only a fraction of their parameters per token. This is the single biggest performance lever on bandwidth-limited hardware.

| Model | Architecture | Active Params | Size | tok/s |
|---|---|---|---|---|
| `btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit` | MoE, 3B active | 3B of 30B | ~8 GB | ~35 |
| `btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit` | MoE, 3B active | 3B of 30B | ~15 GB | ~25 |
| `dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16` | MoE, 3B active | 3B of 80B | ~46 GB | ~18 |
| `Qwen/Qwen3-14B-AWQ` | Dense, 14B | 14B | ~7 GB | ~12 |

The 30B MoE model at 4-bit gets **3x the speed** of the 14B dense model despite having more total parameters, because it only reads 3B parameters per token instead of 14B.

### Avoid Dense Models Above 14B

Dense models (Llama, Mistral, GPT-OSS, Command-R) read every parameter on every token. A dense 70B model at FP16 would need to read 140 GB per token -- that's 0.7 seconds per token at 200 GB/s. Stick to MoE.

---

## Rule #2: Enable TunableOp

Strix Halo's gfx1151 is a new architecture without pre-tuned compute kernels. PyTorch's TunableOp benchmarks kernel variants at runtime and selects the fastest one.

```bash
# Inside the toolbox
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve <model> --enforce-eager --api-key local-dev-key
```

**Impact:** ~2x speedup (e.g. 6 tok/s -> 12 tok/s on Qwen3-14B-AWQ).

The first few requests will be slower as kernels are tuned. Results are cached in `~/.cache/` for subsequent runs.

### For Service Mode

Add the environment variables to `vllm_extra_args` or the Quadlet environment. The tuning happens once and results persist.

---

## Rule #3: Use --enforce-eager

Disable CUDA/HIP graph capture:

```bash
vllm serve <model> --enforce-eager
```

HIP graph capture adds overhead on iGPUs where the kernel launch latency is already low. `--enforce-eager` avoids this overhead and often improves or matches graph-captured performance on Strix Halo.

---

## Rule #4: Kernel Parameters

Enable IOMMU passthrough and expand GPU-accessible memory:

```yaml
# inventory/group_vars/all.yml
strix_halo_kernel_args_enabled: true
strix_halo_kernel_reboot_allowed: true
```

This adds to the kernel command line:

```
iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

| Parameter | Effect |
|---|---|
| `iommu=pt` | IOMMU passthrough -- reduces DMA overhead |
| `amdgpu.gttsize=126976` | ~124 GB GTT (GPU-accessible system RAM) |
| `ttm.pages_limit=32505856` | Raises the TTM page limit to match |

**Requires a reboot** after first application. The role handles this automatically when `strix_halo_kernel_reboot_allowed: true`.

Verify after reboot:

```bash
cat /proc/cmdline | tr ' ' '\n' | grep -E 'iommu|gttsize|pages_limit'
```

---

## Rule #5: Quantization

Lower-bit quantization reduces bytes per parameter, directly increasing tok/s:

| Format | Bits/param | Relative Speed | Quality Impact |
|---|---|---|---|
| FP16 | 16 | 1x (baseline) | None |
| GPTQ-8bit | 8 | ~2x | Minimal |
| GPTQ-4bit / AWQ | 4 | ~4x | Small |
| GPTQ-3bit | 3 | ~5x | Noticeable |

For Strix Halo, **4-bit quantization** (GPTQ-Int4, AWQ) is the sweet spot -- major speed gain with minimal quality loss.

---

## Rule #6: Context Length

Longer context = more KV cache memory = less room for model weights + slower attention.

If you don't need long context, limit it:

```bash
vllm serve <model> --max-model-len 4096
```

Default context lengths (8K-32K) are fine for most use cases. Only increase if you specifically need long-document processing, and expect slower generation.

---

## Putting It All Together

### Maximum Speed (Toolbox Mode)

```bash
toolbox enter vllm

PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit \
  --enforce-eager \
  --api-key local-dev-key
```

Expected: **~35 tok/s** on 128 GB Strix Halo.

### Maximum Quality

```bash
toolbox enter vllm

PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16 \
  --enforce-eager \
  --api-key local-dev-key
```

Expected: **~18 tok/s** on 128 GB Strix Halo. This is the largest high-quality model that fits in 128 GB.

### Balanced (Good Speed + Good Quality)

```bash
toolbox enter vllm

PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit \
  --enforce-eager \
  --api-key local-dev-key
```

Expected: **~25 tok/s** on 128 GB Strix Halo.

---

## What Doesn't Help

| Technique | Why It Doesn't Help on Strix Halo |
|---|---|
| Larger MoE tile sizes (BLOCK_SIZE_N=128) | 2 MB L2 cache is too small; causes cache thrashing |
| Flash Attention v2 | Designed for HBM; limited benefit on system RAM |
| Tensor parallelism | Only 1 GPU -- nothing to parallelize across |
| XCCL / RCCL multi-GPU comms | Single iGPU; XCCL warnings are harmless |

---

## Benchmark Reference

All numbers on AMD Ryzen AI Max "Strix Halo", 128 GB LPDDR5-8000, 8-channel, Fedora 43, ROCm 7.12 (TheROCk), vLLM with `--enforce-eager` and TunableOp enabled.

| Model | Type | Size | tok/s |
|---|---|---|---|
| Qwen3-Coder-30B-A3B (GPTQ-4bit) | MoE | ~8 GB | ~35 |
| Qwen3-Coder-30B-A3B (GPTQ-8bit) | MoE | ~15 GB | ~25 |
| Qwen3-Next-80B-A3B (GPTQ-Int4) | MoE | ~46 GB | ~18 |
| Qwen3-14B-AWQ | Dense | ~7 GB | ~12 |
| Qwen3-14B-AWQ (without TunableOp) | Dense | ~7 GB | ~6 |

---

## Further Reading

- [Getting Started](GETTING_STARTED.md) -- Full setup walkthrough
- [Troubleshooting](TROUBLESHOOTING.md) -- Fix common issues
- [Variables Reference](VARIABLES.md) -- All configuration options
