# Performance Tuning

Maximize inference speed (tok/s) on AMD Strix Halo gfx1151 APUs.

---

## Hardware Overview

Strix Halo is fundamentally different from discrete GPUs:

| Property | Strix Halo (gfx1151) | Discrete GPU (e.g. MI300X) |
|---|---|---|
| **Architecture** | RDNA 3.5 iGPU (gfx1151) | CDNA 3 |
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

## llama.cpp (Vulkan Backend)

The `llamacpp` deployment mode uses the Vulkan backend instead of ROCm/HIP. This is **required** for Qwen 3.5 models which hang on ROCm ([ROCm#6027](https://github.com/ROCm/ROCm/issues/6027)).

### Confirmed Benchmarks (Vulkan)

| Model | Architecture | Active Params | Size | Quant | tok/s |
|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | MoE | 3B | 17 GB | Q4_K_XL | **83.4** |
| Nemotron-3-Nano-30B-A3B | Hybrid Mamba-Transformer MoE | 3B | ~20 GB | Q4_K_XL | **~95** |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | Hybrid Mamba-2 + MoE + Attn | 3B | ~35 GB | Q8_0 | **51.4** |
| Qwen3-Coder-Next-80B-A3B | MoE + hybrid linear attention | 3B | ~85 GB | Q8_0 | **42.7 fresh / 38.1 at 32K** |
| Ling-3.0-flash (124B, `bailingmoe3`) † | MoE + hybrid KDA/MLA | 5.1B | ~73 GB | Q4_K_M | **46.9** |
| Qwen3.5-35B-A3B | MoE | 3B | 21 GB | Q4_K_XL | **59.4** |
| Qwen3.5-122B-A10B | MoE | 10B | 77 GB | Q4_K_XL | **22.8** |
| NVIDIA-Nemotron-3-Super-120B-A12B | Hybrid LatentMoE (Mamba-2 + MoE + Attn) | 12B | ~63 GB | UD-Q3_K_XL | **15.6** |
| MiniMax-M2.7 (229B) | MoE | 10B | ~108 GB | UD-IQ4_XS | **28.1 fresh / 24.5 at 8K / 17.9 at 32K** |

† `Ling-3.0-flash` is **not a profile in this collection**. It requires a community fork of llama.cpp (`aetherbird/llama.cpp`, branch `bailingmoe3-support`) because the `bailingmoe3` architecture is unsupported upstream. Listed here for comparison only; see the expedition report for methodology and caveats.

### Vulkan-Specific Tuning

- **`-b 512`** (batch size, logical): Default. Per-profile override in `llamacpp_model_profiles.<profile>.batch_size` (e.g. `super` and `minimax` both use `4096` for faster prefill on long prompts)
- **`-ub 512`** (ubatch, physical): Default. Set `llamacpp_ubatch_size` globally or per-profile `ubatch_size` — this is the single biggest prefill lever on Vulkan/RADV (gfx1151-specific; decode is unaffected). Measured sweep on the `super` profile with 1.6K-token prompts:

  | `-ub` | Prefill t/s |
  |---|---|
  | 1024 | 169 |
  | 1536 | 283 |
  | **2048** | **311** ← peak |
  | 4096 | 282 ← regresses |

  **2048 is the peak for `super`.** Above it throughput drops — likely iGPU cache thrash or memory-bandwidth saturation on the wider ops in RDNA3.5. The older `1024` recommendation leaves ~45% of prefill throughput on the table.

  > **The optimum is model-dependent, not a gfx1151 constant.** An earlier revision of this document called 2048 "the gfx1151 sweet spot". Subsequent measurement contradicts that: `ub=2048` is the peak for `super` and is worth **+51% decode** on `coder-next`, but it *costs* the `bailingmoe3` (Ling-3.0-flash) architecture ~18% prefill on the same GPU, where the llama.cpp default was better. Sweep per profile rather than copying the value across. `-b` must also be raised alongside `-ub` — llama.cpp clamps the micro-batch to the logical batch, so `-b 512 -ub 2048` silently does nothing.

  Re-verified on hardware at `-ub 2048` (3 cold runs per size, `cache_prompt: false`):

  | Prompt size | Prefill t/s (median) |
  |---|---|
  | 1.6K | 324 (334 peak) |
  | 6K | 357 |
  | 12K | 356 |

  Throughput **plateaus around 6K** and holds flat through 12K — it does not keep climbing with prompt length. Below ~1K tokens, fixed per-request overhead dominates and the effective rate falls off sharply (real traffic shows 70-100 t/s on sub-150-token prompts), which is expected and not a tuning problem.

  Note that `-ub` also sets the compute buffer size (~14.1 GiB at 2048), so it is the main non-weight memory consumer — see the `super` breakdown below.
- **`--no-mmap`**: Required on Strix Halo unified memory — prevents crashes
- **`-fa 1`** (flash attention): Required for stability on gfx1151. Combined with quantized KV cache (both k and v must be the same quant), llama.cpp v4100+ uses a DP4A integer-dot-product fast path for Vulkan (see [PR #20797](https://github.com/ggml-org/llama.cpp/pull/20797))
- **KV cache quantization** (`--cache-type-k q8_0 --cache-type-v q8_0`): Saves ~50% KV memory for longer context. Note: mismatched k/v quants (e.g. k=q8_0 v=q4_0) disable the DP4A fast path — keep them matched

### Sizing `--cache-ram` (prompt-cache pool)

`--cache-reuse 256` gives automatic prefix-KV reuse across requests sharing a ≥256-token prefix — a large win on agent workloads where system prompts and tool schemas repeat. Those snapshots live in an in-memory pool whose size is capped by `--cache-ram`, and **that pool is additive on top of weights + KV**. llama.cpp's default is 8 GB, which is not always safe on a 128 GB unified-memory box.

Size it against **total resident usage before the pool** — weights + KV + compute buffers — not weights alone, and not a figure that already includes the pool:

| Resident before pool | `--cache-ram` | Rationale |
|---|---|---|
| **~100+ GB** (`minimax`: ~108 GB weights, tight even before buffers) | `2048` or smaller | Two OOM incidents traced directly to the 8 GB default — the snapshot-save path had nowhere to grow |
| **~70-80 GiB** (`super`: 58.3 GiB weights + 2.25 GiB KV + 14.1 GiB compute at `-ub 2048` ≈ 73 GiB) | `8192` | Lands at ~81 GiB with the pool, leaving ~44 GiB spare. The wider pool pays for itself: every reused prefix skips a full prefill |

The failure mode is not gradual — the pool grows as snapshots accumulate, so a model that starts fine can OOM-kill the service well into a session. When in doubt on a tight fit, cap low.

`--slot-save-path` is unrelated to this pool: it backs explicit `POST /slots/{id}?action=save|restore` API calls and is never auto-populated by `--cache-reuse`.

### Host-Side Tuning (gpu_tuning role)

The `gpu_tuning` role writes two things to persist across reboots:

- **`power_dpm_force_performance_level=high`** — pins the iGPU at max clocks so Vulkan compute load doesn't hunt DVFS states. Measured **+~7% decode** and a 5-10× reduction in tok/s variance on MoE workloads.
- **`amd_iommu=off` kernel arg** (via `kernel_tuning` role) — disables AMD IOMMU. Measured **+~6% memory bandwidth** on gfx1151 per strix-halo-guide. Trade-off: loses DMA isolation (fine on single-user inference hosts). Override with `strix_halo_kernel_args_remove: []` to keep `iommu=pt` if your deployment needs DMA protection.

### Hybrid Mamba-Transformer Models (Nemotron-3)

The `nemotron` and `super` profiles use NVIDIA's hybrid Mamba-Transformer architectures:

- **Nemotron-3-Nano-30B-A3B** — Mamba blocks + MoE (128 experts, 6 active per token). Coding/agentic focus.
- **NVIDIA-Nemotron-3-Super-120B-A12B** — LatentMoE: Mamba-2 + MoE + Attention layers. Reasoning, planning, agentic tool-use. Runs at the **full native 1,048,576-token context** by default. Multi-Token Prediction (MTP) layers may be present in the GGUF but `--speculative-config` support for this arch is not yet wired up in llama.cpp.

**Requirements:**

- llama.cpp build **≥8351** — fixes a `mamba-base.cpp` assertion that crashes earlier builds ([ggml-org/llama.cpp#20570](https://github.com/ggml-org/llama.cpp/issues/20570))
- `super` runs in ~73 GiB total at UD-Q3_K_XL with the full 1M context. Measured breakdown from the running server:

  | Component | Size | Scales with |
  |---|---|---|
  | Model buffers (Vulkan0 + host + CPU) | 58.3 GiB | — |
  | KV cache, q4_0 | **2.25 GiB** | context |
  | Compute buffers (Vulkan0 + host) | 14.1 GiB | **ubatch**, not context |

  **The KV cache is the small part.** Only **8 of 89 layers** carry KV at all — the rest are constant-state Mamba-2 / MoE — so a 1M-token window costs 2.25 GiB, which is why 1M is essentially free versus 512K. The 14.1 GiB compute buffer is a function of `-ub 2048`, so it's the ubatch setting that dominates non-weight memory here, not the context length. Budget accordingly: raising `-ub` costs memory, extending context barely does.

**Decode throughput (measured, 3 runs per depth, `cache_prompt: false`):**

| Context depth | Decode tok/s |
|---|---|
| ~23 tok | 15.68 |
| ~8K | 15.51 |
| ~32K | 15.46 |

Decode is **flat with context** — a 32K context costs ~1.4% versus an empty one. This follows from the same property that makes 1M context cheap: only 8 of 89 layers do attention, so the per-token cost is dominated by the constant-state Mamba-2 / MoE layers. Contrast `minimax`, a conventional MoE, which drops from ~22 to ~17 tok/s between fresh and 40K context.

The practical consequence: **`super` is the right choice for genuinely long-context work**, even though its fresh-context decode is slower than a comparable dense-attention model. It does not degrade as the conversation grows.

> Earlier revisions of this table listed `~22` tok/s for `super`. That figure was an estimate carried over from the profile's introduction and was never measured; the real value is 15.6.

**Tool calling / reasoning:**

- NVIDIA's official spec (vLLM/SGLang) uses `--tool-call-parser qwen3_coder` plus a custom `super_v3` reasoning-parser plugin. **Neither is supported by `llama-server`** — passing them crashes the server with `invalid argument`.
- The profile uses `--jinja` (so the GGUF's bundled chat template handles tool-call JSON) and `--reasoning-format auto`, which surfaces `<think>` blocks in the dedicated `reasoning_content` response field.
- **Do not add `--special`.** It was tried for `<think>`/`</think>` visibility but leaks `<|im_end|>` into the assistant stream as literal text, which breaks the tool-call parser. `--reasoning-format auto` already exposes the reasoning separately, so it buys nothing.
- NVIDIA mandates **`temperature=1.0`, `top_p=0.95`** across reasoning, tool calling, and chat. The profile reflects this.
- Per-request reasoning controls (`enable_thinking`, `low_effort`) are passed via `chat_template_kwargs` in the OpenAI-compatible API body, e.g. `extra_body={"chat_template_kwargs": {"enable_thinking": true, "low_effort": true}}`.

### Nemotron-3.5-Lightning (lightning profile)

Measured on hardware at `-c 262144` with `parallel_slots: 2`, Q8_0, 3 cold runs per point (`cache_prompt: false`):

| Metric | Value |
|---|---|
| Decode @ ~23 tok ctx | **51.43 tok/s** |
| Decode @ 8K ctx | 51.13 tok/s |
| Decode @ 32K ctx | 49.99 tok/s |
| Prefill @ 1.6K prompt | **1246 t/s** |
| Prefill @ 6K prompt | **1455 t/s** |
| Total resident | **~38 GiB** |
| Live slot config | `n_slots = 2, n_ctx_slot = 131072` |

**Decode is flat with context** (−2.8% from empty to 32K), the same hybrid-architecture property that makes `super`'s 1M window cheap.

**Prefill is ~4× `super`** (1455 vs 357 t/s at 6K) — 3B active params per token versus 12B.

**The Q8 quant costs roughly half the decode speed.** `nemotron` (Nano) does ~95 tok/s at Q4_K_XL; this profile does 51.4 at Q8_0. Decode is memory-bandwidth-bound, and Q8_0 moves ~1.75× the bytes, predicting ~54 tok/s — the measurement matches. The quant was chosen for output fidelity in tool-calling, not throughput. **If the executor tier needs to be faster, move this profile to Q4/Q5** and expect roughly nemotron-class decode.

**Memory:** ~38 GiB total leaves ~87 GiB free, so the 262144 default is conservative and can be raised. It co-resides with `super` (~73 GiB) but **not** with `coder-next` (~85 GiB) or `minimax` (~108 GiB).

**Slots subdivide context.** `-np 2` at `-c 262144` yields a 131072-token window per request, confirmed live. Raise `ctx_size` alongside `parallel_slots` if each request needs the full window.

### Big-coder head-to-head: coder-next vs minimax

Both measured on llama.cpp build 10400, same harness, cold prefill.

| | coder-next (Q8_0) | minimax (UD-IQ4_XS) | delta |
|---|---|---|---|
| Decode, fresh | **42.7** | 28.1 | **+52%** |
| Decode @ 8K | **41.3** | 24.5 | **+69%** |
| Decode @ 32K | **38.1** | 17.9 | **+113%** |
| Prefill @ 6K | **688 t/s** | 316 t/s | **+118%** |
| Resident | **87 GiB** | 110 GiB | 23 GiB smaller |
| Context **per request** | **131072** (262144 ÷ 2 slots) | 81920 (1 slot) | **1.6×** |
| Context aggregate | 262144 across 2 slots | 81920 | 3.2× |

**`coder-next` wins every throughput axis and is smaller.** Caveats worth stating plainly:

- The two ran at different context/slot settings (262144 across 2 slots vs 81920 on 1 slot). Slot count has little effect on single-request decode and allocated context does not change decode rate, so the throughput rows are broadly fair — but it is not a controlled A/B.
- **Read the per-request context row, not the aggregate, when choosing for long sessions.** `parallel_slots: 2` subdivides `ctx_size`, so a single agent conversation on `coder-next` gets 131072 tokens, not 262144. To give one request the full window, set `parallel_slots: 1` or raise `ctx_size` — at 87 GiB resident there is headroom for the latter.
- **Throughput is not the deciding metric.** `minimax` is the empirical incumbent for the big-coder slot on the basis of output quality on real delegated tickets. Speed says `coder-next` deserves the trial; only cost-per-merged-ticket settles it.
- `coder-next` is Q8_0 (near-lossless) against minimax's IQ4_XS, so it is also carrying a quant-fidelity advantage into any quality comparison — which cuts in its favour for tool-call formatting, but means the two are not matched on precision either.

### Qwen3-Coder-Next (coder-next profile)

Measured on hardware at `-c 262144`, `parallel_slots: 2`, Q8_0. Values are medians of the per-request timings llama.cpp logs (n≥2 per point).

Measured on llama.cpp build **10400** with the profile's `batch_size: 4096` /
`ubatch_size: 2048`:

| Metric | Value |
|---|---|
| Decode, fresh ctx | **42.7 tok/s** |
| Decode @ 8K | 41.3 tok/s |
| Decode @ 32K | **38.1 tok/s** |
| Prefill @ 1.6K | 721 t/s |
| Prefill @ 6K | 688 t/s |
| Total resident | **87 GiB** |
| Live slot config | `n_slots = 2, n_ctx_slot = 131072` |

**Q8_0 clears the bar — keep it.** The threshold for falling back to Q6_K (65.6 GB) was 30 tok/s; this sustains 41.3 at 8K. Q6_K remains documented as a fallback but is not needed.

**Batch tuning is mandatory on build 10400.** At the llama.cpp defaults (`b=512, ub=512`) this profile loses about a third of its decode:

| Config | Decode fresh / 8K / 32K | Prefill 1.6K / 6K |
|---|---|---|
| `b512 / ub512` (defaults) | 28.2 / 27.6 / 25.3 | 703 / **814** |
| **`b4096 / ub2048`** (profile) | **42.7 / 41.3 / 38.1** | **721** / 688 |

Tuning buys **+51% decode fresh and +50% at 32K** for −15% prefill at 6K. Decode dominates agentic latency, so the trade is worth taking — but if a workload is overwhelmingly prompt-ingestion, the defaults give better 6K prefill.

**The ubatch optimum is model-dependent, not a property of gfx1151.** `ub=2048` is the peak for this profile and for `super`, but it *costs* the `bailingmoe3` (Ling) architecture ~18% prefill on the same GPU, where the default was better. Re-measure per profile rather than copying the value across.

> **Superseded figures.** Earlier revisions listed 42.3 / 40.9 / **27.5** tok/s and described a **−35%** decode falloff to 32K as an architectural trait of Qwen3-Next's hybrid linear attention, contrasted against the Nemotron profiles. Those numbers were taken on build **8985** at default batch settings. Build-matched at 10400 the falloff is **−11%** (42.7 → 38.1), so the architectural claim was overstated — most of the apparent weakness was toolchain and batch configuration, not attention design.

**Total size does not predict MoE decode speed — active params do.** `coder-next` (80B total) runs at 42.3 fresh against `lightning`'s (30B total) 51.4, despite being 2.4× the weights. Both activate 3B per token, so they land in the same band; the gap is architecture, not scale. Quant width still matters *within* a given active-param count, which is why Q8 vs Q4 moves `lightning` but total-size scaling does not.

**Memory:** 87 GiB at the full 262144 context leaves ~38 GiB free. It does **not** co-reside with `lightning` (~38 GiB) — together they exactly exhaust 125 GiB.

### MiniMax-M2.7 (minimax profile)

MiniMax-M2.7 is a 229B-parameter MoE with 10B active per token, shipped here at `UD-IQ4_XS` (~108 GB). This is the tightest fit in 128 GB unified memory — leave `strix_halo_mode: "llamacpp"` as the only active backend and do not run Open WebUI or other containerized models alongside it.

**Launch params baked into the profile:**

- `--temp 1.0 --top-p 0.95 --top-k 40` (MiniMax-recommended sampling)
- `--jinja` (drives tool calling via the chat template embedded in the GGUF)
- `--reasoning-format auto` (llama.cpp detects the model's reasoning block format)
- `batch_size: 4096`, `ubatch_size: 1024` — aggressive prefill batching (measured ~+30% prefill tok/s on >1K-token prompts vs default `-ub 512`)
- `cache_type_k/v: q4_0` — halves KV footprint and enables the Vulkan DP4A fast path
- `--cache-reuse 256` — automatic prefix-KV reuse across requests that share a ≥256-token prefix (most meaningful on agent workloads where system prompts + tool schemas repeat)
- `--cache-ram 2048` — caps the in-memory prompt-cache pool at 2 GB (default 8 GB). **Critical for MiniMax** — at 128 ctx with q8_0 KV, the default 8 GB cap's snapshot-save path OOM-killed us. q4_0 + 2 GB cap avoids the crash
- `--slot-save-path /models/slot-cache` — directory for explicit `POST /slots/{id}?action=save|restore` calls. Not auto-populated (`--cache-reuse` is RAM-only); reserved for future client-side slot persistence
- `--metrics` — enables Prometheus `/metrics` endpoint for dashboards
- `-np 1` — single request slot; 108 GB model + KV cache leaves no room for concurrency

**Benchmarked on llama.cpp build 10400** (`-c 81920`, 1 slot, UD-IQ4_XS), 3 cold runs per point with `cache_prompt: false`:

| Metric | Value |
|---|---|
| Decode, fresh ctx | **28.14 tok/s** |
| Decode @ 8K | 24.47 tok/s |
| Decode @ 32K | **17.87 tok/s** |
| Prefill @ 1.6K | 325 t/s |
| Prefill @ 6K | 316 t/s |

The earlier hand-observed figures below understated fresh decode (20-22 vs 28.1 measured). Part of that gap is the toolchain: those numbers were taken on build 8985, and this run is on 10400. The deep-context figure held up well (17-18 estimated vs 17.87 measured).

<details><summary>Earlier hand-observed figures (build 8985, kept for comparison)</summary>

| Position | Decode tok/s |
|---|---|
| Fresh (<5K) | 20-22 |
| Mid (15-25K) | 19-20 |
| Deep (35-45K) | 17-18 |
| Cache-reuse hit | 27-28 (effective-zero position) |
| Peak prefill (>10K prompt) | 200-285 tok/s |

</details>

Summary-reset recovers decode by ~3 tok/s — the agent client is responsible for summarizing (llama.cpp doesn't auto-summarize).

> **Tool calling / reasoning parsers:** MiniMax's model card references `--tool-call-parser minimax_m2` and `--reasoning-parser minimax_m2_append_think`. Those are **vLLM** flags and crash `llama-server` with `invalid argument`. In llama.cpp, `--jinja` handles tool calls via the GGUF's chat template and `--reasoning-format auto` handles the reasoning output.

**CUDA 13.2 warning:** [Unsloth's model card](https://huggingface.co/unsloth/MiniMax-M2.7-GGUF) warns that running these GGUFs on CUDA 13.2 produces gibberish output. This deployment uses the Vulkan backend so that path is avoided, but be aware if you repoint the container image at a CUDA build.

### Observability

With `llamacpp_log_disable: false` in inventory + `--metrics` in the profile's extra_args:

- **Journal**: each request emits `prompt eval time = ... tokens per second` (prefill) and `eval time = ... tokens per second` (decode) lines — tail with `journalctl --user -u llamacpp-server -f`
- **`/metrics`** (Prometheus): cumulative counters `llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`, `llamacpp:prompt_seconds_total`, etc. Running-avg gauges `llamacpp:prompt_tokens_seconds` and `llamacpp:predicted_tokens_seconds`
- **`/slots`**: current slot state (processing/idle, `n_decoded`, `n_remain`, active `params`)
- **`/props`**: full server config including `build_info`, `chat_template`, `model_alias`

### Context Size vs. Memory (122B model, 77GB)

| Context | KV Cache | Feasible? |
|---------|----------|-----------|
| 16,384 | ~8-12 GB | Yes (default) |
| 32,768 | ~16-24 GB | Tight but possible |
| 65,536+ | ~32+ GB | Risky |

The 30B/35B models (~20GB) leave ~95GB headroom -- context can be pushed to 128K+ with quantized KV cache.

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

All numbers on AMD Ryzen AI Max+ 395, 128 GB LPDDR5x-8000, Fedora 43.

### llama.cpp (Vulkan GGUF)

| Model | Type | Size | Quant | tok/s |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | MoE, 3B active | 17 GB | Q4_K_XL | **83.4** |
| Nemotron-3-Nano-30B-A3B | Hybrid Mamba-Transformer MoE, 3B active | ~20 GB | Q4_K_XL | **~95** |
| Qwen3.5-35B-A3B | MoE, 3B active | 21 GB | Q4_K_XL | **59.4** |
| Qwen3.5-122B-A10B | MoE, 10B active | 77 GB | Q4_K_XL | **22.8** |
| NVIDIA-Nemotron-3-Super-120B-A12B | Hybrid LatentMoE, 12B active | ~63 GB | UD-Q3_K_XL | **15.6** |
| MiniMax-M2.7 (229B) | MoE, 10B active | ~108 GB | UD-IQ4_XS | **TBD** |

### vLLM (ROCm/TheROCk, --enforce-eager + TunableOp)

| Model | Type | Size | tok/s |
|---|---|---|---|
| Qwen3-Coder-30B-A3B (GPTQ-4bit) | MoE | ~8 GB | ~35 |
| Qwen3-Coder-30B-A3B (GPTQ-8bit) | MoE | ~15 GB | ~25 |
| Qwen3-Next-80B-A3B (GPTQ-Int4) | MoE | ~46 GB | ~18 |
| Qwen3-14B-AWQ | Dense | ~7 GB | ~12 |

---

## Further Reading

- [Getting Started](GETTING_STARTED.md) -- Full setup walkthrough
- [Troubleshooting](TROUBLESHOOTING.md) -- Fix common issues
- [Variables Reference](VARIABLES.md) -- All configuration options
