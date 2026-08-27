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

> **One deliberate exception now ships: the `qwen38` profile (Qwen3.8-27B, dense).** The rule above is about *throughput*, and it holds — at 12.3 tok/s it is the slowest profile in the fleet bar none. It is included anyway because it buys things no MoE here offers at its size: a native vision encoder, a 262K context that costs only ~30 GB resident, and the best capability-per-gigabyte on the box. It also turns out to run at a *higher* fraction of memory bandwidth than any MoE profile (~216 GB/s effective vs `big`'s ~157), because dense weight reads are sequential where expert gathers are scattered. So read the rule as "dense costs you tok/s," not "dense wastes the hardware." See [Qwen3.8-27B (qwen38 profile)](#qwen3827b-qwen38-profile).

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
| Qwen3-Coder-30B-A3B | MoE | 3B | 17 GB | Q4_K_XL | **96.6** |
| Nemotron-3-Nano-30B-A3B | Hybrid Mamba-Transformer MoE | 3B | ~20 GB | Q4_K_XL | **71.1** |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | Hybrid Mamba-2 + MoE + Attn | 3B | ~35 GB | Q8_0 | **56.2** |
| Qwen3-Coder-Next-80B-A3B | MoE + hybrid linear attention | 3B | ~85 GB | Q8_0 | **42.7 fresh / 38.1 at 32K** |
| Ling-3.0-flash (124B, `bailingmoe3`) † | MoE + hybrid KDA/MLA | 5.1B | ~73 GB | Q4_K_M | **46.9** |
| Qwen3.5-35B-A3B | MoE | 3B | 21 GB | Q4_K_XL | **62.1** |
| Qwen3.5-122B-A10B | MoE | 10B | 77 GB | Q4_K_XL | **23.3** |
| NVIDIA-Nemotron-3-Super-120B-A12B | Hybrid LatentMoE (Mamba-2 + MoE + Attn) | 12B | ~63 GB | UD-Q3_K_XL | **18.8** |
| MiniMax-M2.7 (229B) | MoE | 10B | ~108 GB | UD-IQ4_XS | **28.1 fresh / 24.5 at 8K / 17.9 at 32K** |
| Qwen3.8-27B ‡ | **Dense** + Gated DeltaNet hybrid | 27B (dense — all of them) | ~16 GB | UD-Q4_K_XL | **12.4 fresh / 12.3 at 8K / 11.8 at 32K** |

‡ `Qwen3.8-27B` is the only **dense** model in this table and the only one whose tok/s is bounded by weights rather than by architecture choices — see [Qwen3.8-27B (qwen38 profile)](#qwen3827b-qwen38-profile) before comparing it to anything above it.

† `Ling-3.0-flash` is **not a profile in this collection**. It requires a community fork of llama.cpp (`aetherbird/llama.cpp`, branch `bailingmoe3-support`) because the `bailingmoe3` architecture is unsupported upstream. Listed here for comparison only; see the expedition report for methodology and caveats.

### Vulkan-Specific Tuning

- **`-b 4096`** (batch size, logical): Raised from llama.cpp's 512 default. **Must be >= ubatch** — llama.cpp clamps the micro-batch to the logical batch, so `-b 512 -ub 2048` silently runs at 512. Per-profile override via `llamacpp_model_profiles.<profile>.batch_size`.
- **Whole-fleet re-baseline (build 10400, `-b 4096`, llama-bench, r=2).** Every profile was re-measured after the image pin; `-ub 2048` beat the 512 default on long-prompt prefill for all of them, with decode unaffected:

  | Profile | tg128 | pp4096 @ub512 | pp4096 @ub2048 | prefill gain |
  |---|---|---|---|---|
  | `coder` | 96.6 | 1185.5 | 1297.7 | +9% |
  | `fast` | 62.1 | 1104.7 | 1191.6 | +8% |
  | `nemotron` | 71.1 | 1226.8 | **1709.3** | **+39%** |
  | `lightning` | 56.2 | 1202.2 | **1727.2** | **+44%** |
  | `super` | 18.8 | 223.2 | 282.7 | +27% |
  | `big` | 23.3 | 342.7 | 454.8 | +33% |

  This is why `llamacpp_ubatch_size` now defaults to **2048** and `llamacpp_batch_size` to **4096**.

  > **Several published tok/s figures were wrong and are corrected above.** `nemotron` was listed as `~95` and measures **71.1** (−25%) — it was an unmeasured estimate from the same commit as the `super` figure already corrected earlier. `coder` was understated at 83.4 (actual **96.6**), `super` at 15.6 (actual **18.8**), `lightning` at 51.4 (actual **56.2**). Some of the gain is the newer build; the `nemotron` error was never a measurement at all.
- **`-ub 512`** (ubatch, physical): Default. Set `llamacpp_ubatch_size` globally or per-profile `ubatch_size` — this is the single biggest prefill lever on Vulkan/RADV (gfx1151-specific; decode is unaffected). Measured sweep on the `super` profile with 1.6K-token prompts:

  | `-ub` | Prefill t/s |
  |---|---|
  | 1024 | 169 |
  | 1536 | 283 |
  | **2048** | **311** ← peak |
  | 4096 | 282 ← regresses |

  **2048 is the peak for `super`.** Above it throughput drops — likely iGPU cache thrash or memory-bandwidth saturation on the wider ops in RDNA3.5. The older `1024` recommendation leaves ~45% of prefill throughput on the table.

  > **The optimum is architecture-dependent, not a gfx1151 constant.** An earlier revision called 2048 "the gfx1151 sweet spot". Measurement contradicts that, and there is now a clear split:
  >
  > | Architecture family | Best `-ub` | Evidence |
  > |---|---|---|
  > | Qwen3 / Qwen3.5 / Nemotron (`qwen3moe`, `qwen35moe`, `nemotron_h_moe`) | **2048** | +8% to +44% prefill across six profiles |
  > | `bailingmoe3` (Ling-3.0-flash) | **default (512)** | 2048 costs ~18% prefill |
  > | `deepseek4` (DeepSeek-V4-Flash) | **1024** | 2048 costs **22%** prefill (82.1 vs 104.8 t/s at pp4096) |
  > | `qwen35` **dense** (Qwen3.8-27B) | **default (512)** | prefill falls monotonically: 361.0 / 345.1 / 336.1 at 512 / 1024 / 2048 |
  >
  > All three *novel* architectures reject the value that every mainstream one prefers. Sweep per profile before assuming it transfers — it takes one `llama-bench` run. `-b` must also be raised alongside `-ub`: llama.cpp clamps the micro-batch to the logical batch, so `-b 512 -ub 2048` silently does nothing.

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

### Qwen3.8-27B (qwen38 profile)

The fleet's first **dense** model and its first Qwen3.8. Read its throughput with that framing: every other profile here is MoE and reads 2-6 GiB per token, while a dense 27B reads all ~16 GiB. This is the capability/vision/long-context tier.

Measured on build 10400, UD-Q4_K_XL (16.34 GiB), `-b 4096 -ub 512`, `-fa 1`, KV `q8_0`, full 262,144 ctx:

| Depth | `prompt_n` | Prefill t/s | Decode tok/s |
|---|---|---|---|
| fresh | 24 | 87.6 | **12.39** |
| 8K | 5,758 | 360.0 | **12.28** |
| 32K | 23,000 | 314.9 | **11.78** |

`llama-bench` agrees independently: **tg128 12.32**, **pp4096 361.0**.

#### Decode is memory-bandwidth saturated — no *conventional* knob will move it

12.32 tok/s x 16.34 GiB read per token is roughly **216 GB/s of effective weight reads**. Dense inference touches every parameter every token, so that number *is* the memory subsystem running flat out. For contrast, `big` (122B MoE, 10B active) achieves only about 157 GB/s effective — scattered expert gathers have worse locality, so MoE leaves bandwidth unused that a dense model does not.

Two consequences worth internalising before tuning this profile:

1. **No decode-side knob will help.** Batch, ubatch, thread count and cache quant cannot change how many bytes must cross the bus per token. The sweep shows it directly: decode is 12.32 / 12.31 / 12.31 at `-ub` 512 / 1024 / 2048 — flat to three significant figures.
2. **The only real lever is reading fewer bytes per accepted token** — speculative decoding, or a smaller quant. That lever is real and large, but it is not reachable from the upstream-Vulkan configuration this profile ships (below). Read "saturated" as *saturated at one token per weight read* — speculation breaks that assumption, and third-party measurement on this exact hardware clears the ceiling by ~2.4x.

> The hardware table at the top of this document quotes ~200 GB/s for Strix Halo, and the figure above exceeds it. Note that 216 GB/s is an *effective weight-read rate inferred from* `tok/s x model size`, not a bandwidth benchmark. It does suggest the documented ~200 GB/s is conservative — the 256-bit LPDDR5X-8000 config has a 256 GB/s theoretical ceiling — but it should not be quoted as a measured bandwidth number.

#### Long context is nearly free (16 of 65 layers carry KV)

The layout is 16 x (3 x Gated DeltaNet -> 1 x Gated Attention), so only 16 of 65 blocks hold a KV cache. Decode barely degrades with depth, and the **full 262,144-token context measures ~30 GB resident in total** — weights, KV, compute buffers and all:

| Profile | fresh -> 32K | Falloff |
|---|---|---|
| `qwen38` | 12.39 -> 11.78 | **-5%** |
| `coder-next` | 42.7 -> 38.1 | -11% |
| `minimax` | 28.1 -> 17.9 | -36% |

This is the same property that makes `super` affordable at 1M ctx, and it is the strongest practical argument for this profile: it is the cheapest way in the fleet to hold a very large working set.

Sizing note specific to recurrent/hybrid archs: the server also keeps `--ctx-checkpoints` (default **32**) recurrent-state checkpoints **per slot**, serialized fp32 into host RAM. For this model's geometry one checkpoint is ~149.6 MiB ([#27211](https://github.com/ggml-org/llama.cpp/issues/27211)), so the default is ~4.7 GB per slot *on top of* weights, KV and the `--cache-ram` pool. The profile pins `--ctx-checkpoints 8`. The [`--cache-ram` sizing table](#sizing---cache-ram-prompt-cache-pool) does not account for this term.

#### `-ub 512`: the third novel architecture to reject 2048

Swept on build 10400 (`-b 4096`, `llama-bench`, r=2):

| `-ub` | pp4096 | tg128 |
|---|---|---|
| **512** | **361.0** | 12.32 |
| 1024 | 345.1 (-4%) | 12.31 |
| 2048 | 336.1 (-7%) | 12.31 |

Prefill falls monotonically as `-ub` rises. The fleet default of 2048 is actively wrong here, and the llama.cpp default wins.

**A correctness caveat sits next to this value, and it was tested rather than assumed.** [#27237](https://github.com/ggml-org/llama.cpp/issues/27237) reports that this exact architecture emits garbage or early-EOS output on Vulkan at batch 512, and is correct at 1024/4096. That was checked here across three arms — greedy decoding, five known-answer cases plus three 4.3K-token summarisation runs each:

| Arm | Config | Result |
|---|---|---|
| A | `-b 4096 -ub 512` (this pin) | clean |
| B | `-b 512 -ub 512` (the reporter's own config) | **clean — did not reproduce** |
| C | `-b 4096 -ub 2048` (control) | clean |

No garbage, no degenerate repetition, no early EOS in any arm. The reporter ran Windows 11 with an RX 7900 XTX on AMD's proprietary driver; this box is Fedora with gfx1151 on RADV/Mesa. **This is a negative result on our hardware, not evidence the upstream issue is invalid.**

One case did differ: `137 * 4` was answered `548` in arm A and `554` in arms B and C, at temperature 0. That is fluent-and-wrong rather than corrupt — different batch widths change float reduction order, so greedy output is deterministic only within a fixed config. It is a single sample and should not be read as one config being more accurate than another.

#### MTP speculative decoding: blocked upstream on Vulkan, but not blocked in general

The model ships an MTP head, build 10400 has the machinery (`--spec-type draft-mtp`, `-md`, `-ngld`, `--spec-draft-n-max`), and the profile already stages the draft GGUF on disk. It is still deliberately off.

[#27306](https://github.com/ggml-org/llama.cpp/issues/27306) is open against precisely this combination — gfx1151 / Radeon 8060S / 128 GB / Vulkan-RADV, Qwen3.8-27B plus mmproj from Unsloth. With `--spec-type draft-mtp`, long prefill dies in `vk::Queue::submit` with `ErrorDeviceLost` and a compute-ring reset, leaving the process alive as a zombie (`/v1/models` returns 200, `/completion` returns 500). The same argv with MTP off runs past 125k tokens. Qwen3.8's MTP is `!is_mem_shared`, so the draft context is a **second same-width** `llama_decode` after every target ubatch — that extra decode is the trigger, and `GGML_VK_MAX_NODES_PER_SUBMIT=1` does not prevent it.

**That is a statement about upstream llama.cpp on Vulkan/RADV, not about the hardware.** An earlier revision of this document said "blocked on this hardware", which was too broad — the correction matters because this is the profile's one large unexploited lever.

Note also that #27306's failure is specifically *long unique-prefix prefill* ("tens of thousands of tokens"; MTP-off survives past 125k). It is not a blanket failure at every context length, so MTP on Vulkan may work at short-to-moderate context and fall over only on long prefills — the risk profile that matters for a 262K-context profile, but not for every workload.

##### MEASURED HERE: the `qwen38-fp4` profile clears the ceiling

The [`ROCmFPX`](https://github.com/charlie12345/ROCmFPX) fork of llama.cpp adds a `ROCmFP4` runtime tensor format (ggml types 100-106) that stock llama.cpp rejects. [`kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF) publishes measured numbers for this model on a Ryzen AI Max+ 395 / Radeon 8060S / gfx1151 / 128 GB box under ROCm 7.2.4 — the same machine class as ours:

| `--spec-draft-n-max` | decode @8K | acceptance |
|---:|---:|---:|
| off | 13.46 | — |
| 2 | 27.79 | 0.917 |
| **4** | **30.30** | **0.926** |
| 12 | 19.17 | 0.845 |

with perplexity parity against Q4_K_M (5.8877 vs 5.8926 on wikitext-2) at 13.75 GiB, and prefill improving slightly versus their Q4_K_M reference.

**These are third-party figures. Nothing in that table was measured on our box** — the same standard applied to the `Ling-3.0-flash` row above. Read them alongside these caveats:

- Their Q4_K_M reference decodes **10.70** tok/s at 8K; our Vulkan `qwen38` measures **12.28** at the same depth, so our baseline is already ~15% ahead of theirs. The ~2.8x they quote is against *their* baseline, not ours — against ours the honest multiple is closer to **2.4x**.
- Their prefill at 8K is **317.6** (293 with MTP, which costs ~6%); ours is **360.0**. We would give up prefill to gain decode.
- Their context is validated to **65536**; this profile ships **262144**. If long context is the reason to run this model, that is a real regression, not a footnote.
- It requires a **fork** and a different backend. This collection pins an upstream image by digest precisely so throughput is not a function of when you last deployed. The mitigating detail is that the fork is packaged in the toolbox repo we already pin — `rocm-7.14-rocmfpx` / `rocm-7.2.4-rocmfpx` tags — so it is a tag change rather than a build-it-yourself project.
- Their card warns llama.cpp's `--spec-draft-n-max` default is **16**, far down the wrong side of the curve. **Build 10400 reports `default: 3`**, already near their optimum of 3-4 — so verify the default on whatever build you run rather than trusting either card.

**This has now been done, and it works.** The `qwen38-fp4` profile runs that stack on our box. Measured here, greedy, `cache_prompt: false`, medians:

| Workload | `qwen38` (Vulkan) | `qwen38-fp4` (ROCmFP4+MTP) | Speedup |
|---|---:|---:|---:|
| **Code generation** (400 tok) | 12.3 | **29.11** | **2.4x** |
| Prose, fresh | 12.39 | ~22 | 1.8x |
| Prose, 8K | 12.28 | ~21 | 1.7x |
| Prose, 32K | 11.78 | 18.79 | 1.6x |
| Prefill @8K | **360.0** | 341-346 | 0.95x |
| Prefill @32K | **314.9** | 296.1 | 0.94x |

Decode without speculation is content-independent (it is a bytes-per-token bound), so 12.3 is the fair Vulkan baseline for every row.

**Speculative throughput is workload-dependent, and that is the headline caveat.** The gain tracks draft *acceptance*, not the flags. Measured acceptance on this box was **0.59-0.94** on prose (mean accepted length ~3 of 4 drafted) against the quant author's 0.926 measured with `ignore_eos` on a predictable continuation. Code is where it pays: 29.11 tok/s, and the three runs came in at 29.14 / 29.11 / 29.11 — a 0.1% spread. Agentic coding is exactly the predictable-continuation workload speculation is good at, so **the published ~2.8x is reachable on code and roughly 1.6-1.8x is what prose gives you**. Do not quote one number for both.

Quality did not regress: all eight cases in the output-quality probe pass, including `137 * 4 = 548`, which two of the three Vulkan ubatch arms got wrong. Tool calling returns native `tool_calls` with correct JSON.

Three things this profile costs you, all real:

1. **Vision is off.** Multimodal + MTP aborts the server (`server-context.cpp:3192: fatal error`, `ggml_abort`) after the image decodes successfully. Isolated here by rerunning the same quant and projector with the speculative flags removed, which answers image questions correctly — so the trigger is the *combination*. The quant author's published vision test also omits the spec flags. Use `qwen38` for images.
2. **Context drops 262144 -> 65536.**
3. **A fork, off the upstream release train**, and a second image to keep pinned.

###### Context: the full native 262144, validated past the author's 65536

The quant author validates 65536 and lists higher as *not measured* rather than broken. It was measured here and it holds. Memory was never the constraint — the layout is 16 x (3 x Gated DeltaNet -> 1 x Gated Attention), so only **16 of 65 layers carry KV**:

| Depth | `prompt_n` | Prefill t/s | Decode tok/s |
|---:|---:|---:|---:|
| fresh | 24 | 36.1 | 21.12 |
| 8K | 5,758 | 338.6 | 22.29 |
| 32K | 23,000 | 294.6 | 18.04 |
| 100K | 71,840 | 209.9 | 15.02 |

At 71,840 tokens the profile sat at 48 GB of 125 with 76 GB free, and the service stayed healthy — no device-lost, abort, or ring-timeout lines.

**The risk being tested was not memory, it was MTP.** [#27306](https://github.com/ggml-org/llama.cpp/issues/27306) kills the GPU on long unique-prefix prefill precisely because speculative decoding runs a second same-width draft decode after every target ubatch. That report is Vulkan/RADV; this profile is ROCm, so it does not transfer automatically — hence testing at depth rather than assuming. **It does not reproduce here.** If a device-lost ever does appear on a very long prompt, `ctx_size` and the speculative block are the first two suspects, in that order.

Note the decode figures above are *prose* summarisation and so run below the ~30 tok/s this profile reaches on code — draft acceptance is workload-dependent, as always with speculation.

###### `reasoning_effort` defaults to `xhigh`, which returns empty content

Thinking is on by default at `reasoning_effort: "xhigh"`. With a modest `max_tokens` the whole budget goes to reasoning and `content` comes back **empty** — the model looks broken and is not. The chat template accepts **only** `xhigh`, `medium`, `low`; `"none"` raises.

The profile therefore pins a server-side default via `chat_template_kwargs`, and per-request values still override it:

| Setting | reasoning | content |
|---|---:|---|
| server default (`medium`) | 74 chars | `Paris` |
| per-request `{"reasoning_effort": "low"}` | 53 chars | `Paris` |
| per-request `{"enable_thinking": false}` | 0 | `Paris` |

###### `--ipc=host` is mandatory, and the error will send you the wrong way

Without it the model load dies in the HSA runtime:

```
Memory critical error by agent node-0 ... Reason: Memory in use.
(libhsa-runtime64.so / libggml-hip.so)
```

That reads like an OOM or a locked-memory problem. It is neither. ROCm's HSA runtime allocates through shared-memory segments, and rootless podman's private IPC namespace breaks it. Isolated on this box, minimal load, one variable at a time:

| Container flags | Result |
|---|---|
| none | HSA critical error |
| `--security-opt seccomp=unconfined` | HSA critical error |
| **`--ipc=host`** | **loads cleanly** |

Necessary and sufficient. Note this box's `RLIMIT_MEMLOCK` is 8 MB soft *and* hard (`DefaultLimitMEMLOCK`), so neither rootless podman nor a systemd user unit can raise it — and it is **not** the cause, so no root change is needed. The error message strongly invites that detour; do not take it.

**The `ngram-*` `--spec-type` variants** allocate no draft context at all, so they should sidestep #27306's code path while staying on upstream Vulkan. Unmeasured here, and the cheapest of the three experiments.

#### Vision

Native VLM. Vision is silently **off** unless the projector is passed: the server starts normally and simply rejects image parts. The profile sets `mmproj_file: mmproj-F16.gguf`, which the role renders as `--mmproj`. Verified end-to-end on a synthetic 224x224 image (yellow disc on blue), answered correctly at 90 prompt tokens.

### DeepSeek-V4-Flash-0731 (deepseek-v4 profile)

A 284B MoE / 13B active running on a 128 GB APU, at 2.90 bpw in 97 GiB. The capability flagship of the fleet — it reports Terminal Bench 2.1 **82.7** against Qwen3.8-27B's 73.0, on a model that fits.

Measured here on mainline llama.cpp build 10217 for ROCm/HIP, ctx 32768, code generation, median of 3:

| Config | Image | GTT | Decode | Prefill |
|---|---|---:|---:|---:|
| **No drafter, all on GPU** | **mainline 10217** | 97 GiB | **17.07** | ~37-40 |
| No drafter, all on GPU | ROCmFPX fork (211) | 97 GiB | 12.22 | ~34 |
| No drafter, `--n-cpu-moe 16` | ROCmFPX fork | 65 GiB | 10.74 | 26.3 |
| DSpark drafter + `--n-cpu-moe 16` | ROCmFPX fork | 77 GiB | 4.22 | 7.5 |

Prefill scales steeply with prompt length, so a short-prompt number badly understates it:

| Prompt | Prefill t/s | Decode tok/s |
|---:|---:|---:|
| 24 tok | ~37 | 17.16 |
| 5,594 | **161.8** | 15.97 |
| 22,370 | **141.6** | 14.89 |

That **matches or beats the author's ~130 t/s**, and their ~14-16 t/s at 16k. An earlier revision of this document claimed our prefill was "~3.5x short and unexplained"; that was measured on a 24-token prompt where fixed per-request overhead dominates, and it was wrong. The same shape appears on `qwen38` (87.6 t/s at 24 tokens, 360 at 5,758) — never characterise prefill from a short prompt.

#### Context is almost free — the full 1M window fits

MLA makes this the cheapest long context in the fleet, and the naive arithmetic is badly misleading. From the GGUF geometry — `head_count_kv = 1`, key/value length 512, 43 blocks — you would derive ~45.7 KiB/token at q8_0. **Measured, it is ~5.3 KiB/token**: GTT grew 115 MiB across a 22,370-token prompt. llama.cpp stores MLA's *compressed latent*, not the nominal 512+512, so the derivation is off by ~9x.

| ctx | KV | GTT total | Decode @8K | Prefill @5.6K |
|---|---:|---:|---:|---:|
| 65536 | ~0.3 GiB | 97 GiB | — | — |
| 262144 | ~1.3 GiB | 97 GiB | 15.95 | 161.5 |
| **1048576** (shipped) | ~5.3 GiB | **101 GiB** | **15.97** | **161.8** |

Sixteen times the window for ~4 GiB of GTT and no measurable throughput cost. Note also that llama.cpp allocates KV **lazily** — GTT at load is identical for 65536 and 262144, so `ctx_size` alone tells you nothing about what a profile will actually consume. Watch `mem_info_gtt_used` under real traffic.

Two caveats, neither about memory:

- **65536 is the trained window; 1M is extrapolation.** rope scaling is `yarn`, factor 16.0, `original_context_length` 65536 — and 65536 x 16 = 1048576 exactly. Reaching past 65536 trades quality for reach by design.
- **Prefill is the real ceiling.** At ~140-160 t/s, filling 1M tokens takes roughly two hours. The window is there when a task needs it; it is not one you fill casually.

#### Four findings that contradict or extend the artifact's card

1. **The ROCmFPX fork costs 40% of decode on this model.** 12.22 vs 17.07 for the identical quant and flags. The fork is mainline plus extra ROCmFP4 tensor types and loads this model correctly, so substituting it looks free — it is not. `qwen38-fp4` genuinely needs the fork (its quant uses those tensor types); this profile must not use it.
2. **The DSpark drafter is a net loss, then unloadable.** On the fork it costs 2.5x (4.22 vs 10.74 at equal `--n-cpu-moe`), because the drafter is 10.15 GiB while the target reads only ~4.7 GB per token at 13B active — every drafted token costs more than generating one. Acceptance was healthy at 0.82-0.84, so the speculation works; it just cannot pay for its own weight. On mainline it will not load at all: `key not found in model: dflash.attention.sliding_window_pattern`. Compare `qwen38-fp4`, where a small MTP head pays 2.4x — **drafter size relative to per-token target bytes is what decides it**, not the flags.
3. **`--n-cpu-moe 16` costs 12%, not 0.1%.** 10.74 vs 12.22 on the fork. The full 97 GiB fits in the 124 GiB GTT, so the offload buys nothing here. GTT by setting: 16 -> 65 GiB, 8 -> 81 GiB, none -> 97 GiB.
4. **Two silent misconfigurations cost 3-4x each, with no error in the log.** Both are worth knowing because the only symptom is throughput:
   - `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` (correct for `qwen38-fp4`, wrong here) makes HIP allocate managed memory that never enters GTT. `mem_info_gtt_used` sat at **0 of 124 GiB** while a 97 GiB model ran from host memory: 3.4-3.9 tok/s.
   - `n_parallel` left on auto picked **4 slots** at 262144 ctx, pinning 120 of 125 GB and collapsing buff/cache to ~1 GB: 3.88 tok/s decode, 9.4 t/s prefill.

   **`mem_info_gtt_used` is the diagnostic**, not `free`. Check it after every change to this profile.

#### Operational notes

- **Thinking is on by default at `reasoning_effort: xhigh`.** With a small `max_tokens` the entire budget goes to reasoning and `content` returns empty — the model looks broken and is not. The template accepts only `xhigh`, `medium`, `low`; `"none"` throws. Use `chat_template_kwargs: {"enable_thinking": false}` to disable.
- **Wait for GTT release between profile switches.** The driver frees buffers after the process exits, so poll `mem_info_gtt_used` rather than sleeping — starting a 97 GiB load against stale GTT is what triggers the autofit failure above.
- Integrity was verified before first launch (`sha256 538ec693...`), which the card recommends and which is cheap next to a 97 GiB download.

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
| MiniMax-M2.7 (229B) | MoE, 10B active | ~108 GB | UD-IQ4_XS | **28.1 fresh / 17.9 at 32K** |

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

---

## Lemonade Server (11.8.0)

A second inference stack, deployed with `mise run deploy:lemonade`. Measured on the same box, build as pinned in `lemonade_service` defaults, ROCm via TheRock 7.14.0, amdgpu pinned to `high`.

### Qwen3.8-27B — Lemonade/ROCm vs the `qwen38` profile on Vulkan

Same GGUF (`unsloth/Qwen3.8-27B-GGUF`, UD-Q4_K_XL), same box, different backend:

| Path | Decode | ttft | Notes |
|---|---:|---:|---|
| `qwen38` profile (Vulkan, RADV) | 12.3 | — | bandwidth-saturated, no conventional knob moves it |
| **Lemonade `llamacpp:rocm`** | **17.4 - 18.5** | 0.35 - 0.47 s | **1.4 - 1.5x** |

The gain is not the ROCm backend on its own — **Lemonade turns on MTP speculative decoding automatically** for this model (the catalog entry carries an `mtp` label). Measured draft acceptance was **0.48 - 0.51**, mean accepted length 2.4 - 2.5 of the drafted tokens.

Two things follow. First, this is the same lever the `qwen38-fp4` profile uses, but reached without the ROCmFPX fork and without giving up vision. Second, acceptance here is markedly lower than the 0.93 that profile sees on code, so the multiple is correspondingly smaller — 1.4x rather than 2.4x. Acceptance is workload-dependent, and these figures came from short prose prompts.

**Prefill is not characterised.** The only prefill numbers observed were on 17 - 21 token prompts (≈61 t/s), and this document's own rule applies: never characterise prefill from a short prompt. Treat prefill as unmeasured on this path.

### DeepSeek-V4-Flash on ds4 (DwarfStar)

`DeepSeek-V4-Flash-IQ2XXS-DS4`, 81 GB at roughly 2.3 bpw, ctx 32768.

| Source | Decode | Measured by |
|---|---:|---|
| **Lemonade + ds4, warm** | **12.8 t/s** | **here** |
| Lemonade + ds4, `--ssd-streaming-cache-experts 96GB` | 11.9 t/s | here |
| ds4 driven directly, upstream's own figure | 16.45 t/s | [PR #3047](https://github.com/lemonade-sdk/lemonade/pull/3047) — **not measured here** |
| `deepseek-v4` profile (llama.cpp/ROCm, 2.90 bpw) | 17.1 t/s | here, different quant |

**We are ~25% below upstream's number for the same engine and quant, and below the llama.cpp path this collection already had.** The most likely cause is identified and is not a misconfiguration:

#### Lemonade forces `--ssd-streaming`, and it cannot be turned off

`Ds4Server::build_args` appends `--ssd-streaming` unconditionally, on this reasoning:

> ds4-server defaults to full residency, which maps the entire model into the ROCm arena. [...] the only supported device (gfx1151) tops out around a 64 GB VRAM carveout with a smaller usable arena, so full residency always OOMs mid-load.

**That premise does not hold on this box.** This fleet routinely runs fully resident models well past 64 GB — `minimax` at ~108 GB, `deepseek-v4` at ~97 GB — because the working pool is the 124 GB GTT, not a VRAM carveout. Upstream's own 16.45 t/s was measured with full residency (their notes record a 21.4 s load into 100 GiB of GTT); ours streams experts from disk.

`ds4-server` publishes no flag to disable streaming once it is on, and Lemonade inserts the flag before any `ds4_args`, so **there is no way to reach full residency through Lemonade**. Raising the resident expert budget does not substitute: `--ssd-streaming-cache-experts 96GB` was accepted, lifted GTT from ~7.6 GB at load to 72.7 GB, and made decode marginally *worse* (11.9 vs 12.8).

Observed behaviour is consistent with streaming throughout: load reports success in **1.2 s** rather than upstream's 21.4 s, GTT sits at ~7.6 GB immediately after load, and grows to 88 - 107 GB only as generation touches experts.

#### Operational notes

- **Telemetry reports zero for this backend.** Lemonade logs `ttft=0.00s, tps=0.00` on every ds4 completion, and the UI's rate counters follow. The `usage` block in the API response *is* correct (`prompt_tokens`, `completion_tokens`, `total_tokens`). This is the visible face of the known upstream limitation that ds4 streaming is bursty — deltas arrive in one burst at the end, so Lemonade never observes a first token and cannot compute either figure. Real throughput is only in `ds4-server`'s own stdout (`avg=12.8 t/s`), which reaches `journalctl --user -u lemonade-server`.
- **The reasoning/content split leaks.** A raw `</think>` was observed inside `content`, with the answer duplicated on either side of it: `'Paris is the capital of France.</think>Paris is the capital of France.'` It is intermittent — repeat runs of the same prompt came back clean. Because DeepSeek frequently reasons in Chinese, a mis-split surfaces as Chinese text in the answer. Passing `chat_template_kwargs: {"enable_thinking": false}` produced clean output in testing, though `reasoning_content` was still populated, so it is a mitigation rather than a confirmed fix.
- **`max_loaded_models` is 1, so one long generation blocks the whole server.** This is the single most surprising operational property of this stack, and it is not a defect. A realistic UI prompt — "build me a single-page app" — generates tens of thousands of tokens, and at ~12 t/s that is **tens of minutes of wall clock**. One such request was observed running 18,550 tokens over 26 minutes. Throughout, every other client sees connection timeouts and `/api/v1/health` reports `is_busy: true`, which is very easy to misread as a hung server. It is not: `max_tokens` is honoured exactly (`finish_reason: stop`), and unbounded requests terminate normally.

  The practical consequences are worth stating plainly, because the recovery for "hung" and the recovery for "busy" are opposites:

  - **Check before restarting.** `podman logs --tail 5 lemonade-server` shows a live `gen=N ... t/s` counter while ds4 is generating. A rising `gen=` means it is working, and restarting throws away real work — as happened once here.
  - **Do not benchmark against a shared server.** Queued requests inherit the wait of whatever is ahead of them, which silently corrupts any timing measured from the client side.
  - Model-switching contends for the same slot, so a `lemonade load` issued during a long generation waits too.

#### What has not been tried

`--power` is already at its default maximum of 100 and the GPU is pinned to `high`, so neither is a candidate. `--prefill-chunk` (default 4096) affects prefill, which is unmeasured here. `--batched-session` and `--threads` are untested. None of these address residency, which remains the leading hypothesis.
