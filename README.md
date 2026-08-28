# AMD Strix Halo LLM Automation

**nerdsrun.strix_halo_vllm** -- Ansible collection for deploying LLM inference on AMD Ryzen AI Max "Strix Halo" (gfx1151) APUs

![CI](https://img.shields.io/github/actions/workflow/status/nerdsrun/amdllmv/ci.yml?branch=main&label=CI)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

Enterprise-quality, idempotent Ansible automation that takes a Fedora system with AMD Strix Halo hardware from zero to a fully operational LLM inference server. Supports two backends: **vLLM** (ROCm) and **llama.cpp** (Vulkan). Includes toolbox mode for interactive development, service mode for persistent API endpoints, model weight prefetching, and an optional Open WebUI chat frontend.

**Upstream images:** [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes/) (Vulkan/llama.cpp) | [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes/) (ROCm/vLLM)

---

## Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Fedora 43+ |
| **Hardware** | AMD Ryzen AI Max "Strix Halo" APU (gfx1151) |
| **RAM** | 64GB minimum, 128GB recommended |
| **Devices** | `/dev/kfd` and `/dev/dri` present |
| **Task Runner** | [mise](https://mise.jdx.dev/) |
| **Disk Space** | 50 GB minimum; 200 GB+ for full model set |

See [Getting Started](ansible_collections/nerdsrun/strix_halo_vllm/docs/GETTING_STARTED.md) for the full setup walkthrough.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/nerdsrun/amdllmv.git && cd amdllmv

# 2. Bootstrap (installs Ansible, linters, molecule)
mise run bootstrap

# 3. Configure
cp inventory/hosts.yml.example inventory/hosts.yml
$EDITOR inventory/hosts.yml            # set your target host/user
$EDITOR inventory/group_vars/all.yml   # tune deployment options

# 4. Deploy
mise run deploy:toolbox           # interactive toolbox (vLLM/ROCm)
mise run deploy:service           # systemd vLLM API server (ROCm)
mise run deploy:llamacpp          # llama.cpp Vulkan — 122B model (default)
mise run deploy:llamacpp:coder    # llama.cpp — Coder 30B profile
mise run deploy:llamacpp:fast     # llama.cpp — Fast 35B profile
mise run deploy:llamacpp:nemotron # llama.cpp — Nemotron Nano 30B profile
mise run deploy:llamacpp:lightning # llama.cpp — Nemotron 3.5 Lightning 30B profile
mise run deploy:llamacpp:super    # llama.cpp — Nemotron Super 120B profile
mise run deploy:llamacpp:coder-next # llama.cpp — Qwen3-Coder-Next 80B profile
mise run deploy:llamacpp:minimax  # llama.cpp — MiniMax-M2.7 229B profile
mise run deploy:llamacpp:qwen38     # llama.cpp — Qwen3.8-27B dense VLM, 262K ctx (Vulkan)
mise run deploy:llamacpp:qwen38-fp4 # llama.cpp — Qwen3.8-27B ROCmFP4 + MTP, ~2.4x decode (ROCm)
mise run deploy:llamacpp:deepseek-v4 # llama.cpp — DeepSeek-V4-Flash 284B, 1M ctx (ROCm)
mise run deploy:llamacpp:deepseek # llama.cpp — DeepSeek-V4-Flash on Vulkan (gated, see note)
mise run deploy:all               # full site.yml deployment

# 5. Verify
mise run verify

# 6. (Optional) Chat UI
mise run ui:up            # Open WebUI at http://localhost:3000
```

---

## Modes

| Mode | Variable | Backend | Description |
|---|---|---|---|
| `toolbox` | `strix_halo_mode: "toolbox"` | vLLM/ROCm | Interactive toolbox container for development |
| `service` | `strix_halo_mode: "service"` | vLLM/ROCm | Persistent vLLM server via systemd Quadlet (port 8000) |
| `both` | `strix_halo_mode: "both"` | vLLM/ROCm | Deploy both simultaneously |
| `llamacpp` | `strix_halo_mode: "llamacpp"` | llama.cpp/Vulkan | GGUF model server via systemd Quadlet (port 8080) |
| `lemonade` | `mise run deploy:lemonade` | Lemonade (ROCm) | Multi-model **router** via systemd Quadlet (port 13305) — see [Lemonade](#lemonade-server-multi-model-router) |

### llama.cpp Model Profiles

When using `llamacpp` mode, select a model profile with `llamacpp_model_profile` or via mise:

| Profile | Model | Size | Active Params | tok/s | Use Case |
|---|---|---|---|---|---|
| `big` (default) | Qwen3.5-122B-A10B | 77 GB | 10B | **23.3** | Reasoning, vision, general |
| `coder` | Qwen3-Coder-30B-A3B | ~20 GB | 3B | **96.6** | Coding, tool-use, agentic |
| `fast` | Qwen3.5-35B-A3B | ~20 GB | 3B | **62.1** | Fast general + vision |
| `qwen38` | Qwen3.8-27B (dense, native VLM) | ~16 GB | 27B (dense) | **12.4 fresh / 11.8 at 32K** ‡ | Vision + 262K context, best capability-per-GB — **capability tier, not a speed tier** |
| `qwen38-fp4` | Qwen3.8-27B ROCmFP4 + MTP | ~14 GB | 27B (dense) | **29.1 on code / ~22 on prose** ‡‡ | Same model at 2.4x decode via speculative decoding — ROCmFPX fork, **full 262K ctx**, no vision |
| `nemotron` | Nemotron-3-Nano-30B-A3B | ~20 GB | 3B | **71.1** | Coding, agentic, reasoning |
| `lightning` | NVIDIA-Nemotron-3.5-Lightning-30B-A3B | ~35 GB | 3B | **56.2** | Fast agentic executor — tool use, code review, high-volume delegation |
| `super` | NVIDIA-Nemotron-3-Super-120B-A12B | ~63 GB | 12B | **18.8** | Reasoning, planning, tool-calling (1M ctx native) |
| `coder-next` | Qwen3-Coder-Next-80B-A3B | ~85 GB | 3B | **42.7 fresh / 38.1 at 32K** | Agentic coding delegation worker (challenger to minimax) |
| `minimax` | MiniMax-M2.7 (229B MoE) | ~108 GB | 10B | **28.1 fresh / 17.9 at 32K** | Long-context agentic, tool-use, reasoning |
| `deepseek-v4` | DeepSeek-V4-Flash-0731 (284B MoE) | ~97 GB | 13B | **17.1 fresh / 14.9 at 32K** | Capability flagship — Terminal Bench 2.1 82.7 vs Qwen3.8-27B's 73.0. ROCm/HIP, **full 1M ctx** |

> **Note:** The `nemotron` and `super` profiles require llama.cpp build **≥8351** (fixes [ggml-org/llama.cpp#20570](https://github.com/ggml-org/llama.cpp/issues/20570) — mamba-base.cpp assertion crash). Both are hybrid Mamba-Transformer architectures. The `super` profile runs at the **full native 1,048,576-token context** in ~73 GiB total on a 128 GB Strix Halo. The hybrid LatentMoE design means only 8 of 89 layers carry KV at all (the rest are constant-state Mamba-2 / MoE), so a 1M-token window costs just 2.25 GiB of KV cache — 1M is effectively free versus 512K. Non-weight memory is dominated by the ~14 GiB ubatch compute buffer instead, leaving ~50 GiB spare. Tool calling rides on `--jinja` + the GGUF's chat template; the vLLM-only `--tool-call-parser qwen3_coder` / `--reasoning-parser super_v3` flags are NOT supported by `llama-server` and will crash it.
>
> **Nemotron-3.5-Lightning (`lightning`):** Released 2026-08-11; the successor family to the `nemotron` (Nano) profile, same 30B-A3B size class. Hybrid Mamba-2 + MoE + Attention (GGUF arch `nemotron_h_moe`), up to 1M native context, [OpenMDW-1.1](https://openmdw.ai/license/1-1/) license. **Build requirement:** the arch is present in llama.cpp build **8985**, which the pinned `kyuz0/amd-strix-halo-toolboxes:vulkan-radv` image ships — verified by inspecting the image's `libllama.so`, so no gate beyond the existing ≥8351 note applies. Ships at **Q8_0 (35.0 GB)**: near-lossless, and at this size the 128 GB box has abundant headroom, so there is no reason to quantize harder. Context defaults to **262144**. **Measured on hardware:** at `-c 262144` with 2 slots the whole thing sits in **~38 GiB**, leaving ~87 GiB free — so 262K is conservative and there is ample room to raise it. Parallel slots default to **2** via the per-profile `parallel_slots` variable; confirmed live as `n_slots = 2, n_ctx_slot = 131072` — **slots subdivide `ctx_size`**, so two slots at 262K give each request a 131K window. Raise `ctx_size` alongside slots if you need the full window per request.
>
> At ~38 GiB it comfortably co-resides with `super` (~73 GiB) — but **not** with `coder-next` (~85 GiB) or `minimax` (~108 GiB), which would exceed 125 GiB together.
>
> ‡ **`qwen38` is the fleet's only dense model, and its tok/s must be read differently.** Every other profile is MoE and reads 2-6 GiB per token; a dense 27B reads all ~16 GiB. At 12.3 tok/s that works out to roughly **216 GB/s of effective weight reads — the memory subsystem running flat out**, and a *higher* bandwidth utilisation than any MoE profile here achieves (`big` manages about 157 GB/s). It is not slow because it is untuned in the ordinary sense — no batch, ubatch, thread or cache setting moves it, because none change bytes-read-per-token. **Speculative decoding does**, by producing several tokens per weight read, and third-party measurement on this exact machine class reports ~30 tok/s that way. That path needs a llama.cpp fork and a different backend and is not what this profile ships — see [MTP speculative decoding](ansible_collections/nerdsrun/strix_halo_vllm/docs/PERFORMANCE.md#mtp-speculative-decoding-blocked-upstream-on-vulkan-but-not-blocked-in-general). As shipped, pick this profile for vision, 262K context, and capability per gigabyte — not for throughput.
>
> ‡‡ **`qwen38-fp4` is the same model made fast, and the speedup depends on what you ask it.** Speculative decoding (MTP) emits several tokens per weight read, which is the only way past the bandwidth ceiling that caps `qwen38` at 12.3. The gain tracks draft *acceptance*: measured **29.11 tok/s on code generation** (2.4x) but only ~21 on novel prose (1.7x), because code is the predictable-continuation workload speculation is good at. It costs two things — vision is off (multimodal + MTP aborts the server), and it needs the [ROCmFPX](https://github.com/charlie12345/ROCmFPX) fork of llama.cpp rather than the pinned upstream image. Context is **not** one of them: it runs the same full native 262144 window as `qwen38`, validated on hardware to a 71,840-token prompt with no fault. Prefill is marginally *worse* than `qwen38` (339 vs 360 at 8K). Pick `qwen38` for images; pick `qwen38-fp4` for everything else.
>
> **On the Q8 choice and speed:** measured decode is **56.2 tok/s** against `nemotron`'s **71.1** — a 21% gap, not the ~50% an earlier revision of this note claimed. That earlier analysis was anchored on a `~95` figure for `nemotron` that turned out to be an unmeasured estimate; re-benchmarked, `nemotron` is 71.1. Bandwidth alone would predict worse: Q8_0 here is 32.59 GiB against `nemotron`'s 21.26 GiB (1.53×), implying ~46 tok/s. Lightning delivers 56.2, i.e. **~21% better than its byte count predicts** — the 3.5 architecture is more efficient per byte than Nano, so the quant is costing less than it appears. **A Q4/Q5 lightning is still worth trying if the executor tier needs speed**, but temper expectations: the realistic ceiling is Nano-class (~71), not the ~95 previously implied. Prefill is excellent — **1727 t/s at 4K prompts** with `-ub 2048` (up 44% from the llama.cpp default), roughly 6× the `super` profile, because only 3B params are active per token.
>
> **Qwen3-Coder-Next (`coder-next`):** 80B total / 3B active, MoE with hybrid linear attention (GGUF arch `qwen3next`), 256K native context. Published as **`unsloth/Qwen3-Coder-Next-GGUF`** — there is no size suffix in the repo name. Ships at **Q8_0 (84.8 GB, 3 shards)**: near-lossless, chosen deliberately over IQ4 because this profile is a tool-calling delegation worker where the failure mode that matters is quant-induced malformed tool-call JSON, not a benchmark point. **Measured: 42.7 tok/s fresh, 41.3 at 8K — Q8 clears the 30 tok/s bar, so it stays.** Q6_K (65.6 GB) remains documented as a fallback but is not needed. **Context falloff is mild:** decode holds **38.1 tok/s at 32K (−11%)**. Prefill is 688 t/s at 6K. **Requires `batch_size: 4096` / `ubatch_size: 2048`** — at llama.cpp defaults on build 10400 it loses ~33% decode. Resident footprint is **87 GiB** at the full 262144 context, which fits with ~38 GiB spare but does **not** co-reside with `lightning`. Context defaults to the **full native 262144** — hybrid linear attention keeps KV cheap and ~25 GB of post-weights headroom holds it. Parallel slots default to **2** (`parallel_slots`), tunable to 4 for multi-agent delegation. **Note that slots subdivide the window: at 2 slots each request gets 131072 tokens, not 262144** (confirmed live as `n_ctx_slot = 131072`). For a single agent that needs the full 256K, set `parallel_slots: 1` or raise `ctx_size`. **This model is non-thinking** — its card states it never emits `<think>` blocks, so `--reasoning-format auto` is a no-op here and is kept only for consistency. **~85 GB resident + KV — do not run alongside other models.** This profile is a *challenger* to `minimax`, which remains the incumbent big coder until A/B'd on real tickets.
>
> **MiniMax-M2.7:** Ships at UD-IQ4_XS (~108 GB) — the tightest fit on 128 GB unified memory. Do not run alongside other models. Tool calling rides on `--jinja` + the chat template baked into the GGUF, with `--reasoning-format auto` to surface the model's reasoning blocks (llama.cpp does not support the vLLM-style `--tool-call-parser` / `--reasoning-parser` flags referenced in the MiniMax docs). Per [Unsloth docs](https://huggingface.co/unsloth/MiniMax-M2.7-GGUF), do **NOT** run these GGUFs on CUDA 13.2 (produces gibberish) — this deployment uses Vulkan, so no action needed.
>
> **DeepSeek-V4-Flash.** 284B total / 13B active, MoE with hybrid sparse attention (GGUF arch `deepseek4`), 1M native context.
>
> **`deepseek-v4`.** Runs on **ROCm/HIP**, where the arch is fully supported. Ships Kevletesteur's requantization of the Unsloth UD-IQ3_XXS — 103 GB at 2.90 bpw, with the three attention tensor families that make up 51.9% of per-token bytes lifted Q8_0 → Q6_K. Measured here: **17.1 tok/s** fresh, 15.0 at 32K, prefill 141-162 t/s on real prompts, at the **full 1,048,576-token window** for ~101 GiB of GTT. Verified 90.8% token-identical against the full-precision API over 17,929 positions, 240/240 paired-QA parity.
>
> **A second `deepseek` profile targeting Vulkan was removed.** It existed to hold the same model against the default Vulkan image, where `deepseek4` has no implementation, and it was gated so nobody downloaded ~104 GB for a server that cannot load. Once `deepseek-v4` proved the arch runs fine on ROCm, the profile had no remaining purpose: the answer is *use a different backend*, and that is what `deepseek-v4` is. Keeping a permanently-gated profile around only invited someone to ungate it.
>
> Two findings from bringing `deepseek-v4` up are worth carrying: the **DSpark drafter is a net loss** here (the 10.15 GB drafter exceeds the ~4.7 GB the target reads per token — speculation works, at 0.82-0.84 acceptance, but cannot pay for its own weight), and **`--n-cpu-moe` costs 12%, not the 0.1% its card reports**. See [PERFORMANCE.md](ansible_collections/nerdsrun/strix_halo_vllm/docs/PERFORMANCE.md#deepseek-v4-flash-0731-deepseek-v4-profile).

---

## Lemonade Server (multi-model router)

`mise run deploy:lemonade` deploys [Lemonade Server](https://lemonade-server.ai) **11.8.0** as a second, independent inference stack. It is not another llama.cpp profile — it is a *router*: one systemd unit that loads and evicts models on demand across engines, behind one OpenAI-compatible endpoint on port **13305** (`/v1`, `/api/v1` and `/v0` all answer).

| Model | Source | Decode | Notes |
|---|---|---:|---|
| `Qwen3.8-27B-GGUF` | downloaded (17.2 GB) | **18.5 - 27.0 tok/s** | vision + MTP speculative decoding, both from the catalog entry |
| `deepseek-v4` | **already on disk** (0 GB added) | **17.1 tok/s** | alias for the `deepseek-v4` profile's own 2.90 bpw quant, served through Lemonade |

Both run on **llama.cpp/ROCm** (b10469 + TheRock 7.14.0). Only one inference stack can hold the GPU, and that is now enforced by systemd rather than by convention: the Quadlet carries `Conflicts=llamacpp-server.service` and `Conflicts=vllm-server.service`, so starting either one stops Lemonade and vice versa. The role also stops them at deploy time and waits for GTT to drain, because `Conflicts=` releases the unit but the driver frees GTT only when the process exits.

### Why this is faster than the equivalent profiles

**Qwen3.8-27B is 1.5 - 2.2x faster here than the `qwen38` profile serving the identical GGUF** — 18.5-27.0 tok/s against 12.3. Prefill is unchanged (360-364 t/s at 7.7-9.1K prompts, matching the Vulkan profile exactly), so it is purely a decode win. The cause is not the ROCm backend: Lemonade reads the `mtp` label on the catalog entry and turns on **speculative decoding automatically**. That is the same lever `qwen38-fp4` pulls, but without the ROCmFPX fork and **without giving up vision**. Throughput varies with draft acceptance, hence the range.

### `extra_models_dir` — the GGUF tree is mounted, not copied

Lemonade stores its own downloads in HuggingFace cache layout and cannot read the flat `--local-dir` tree `llamacpp_service` builds. Rather than keep two copies, the role bind-mounts `llamacpp_model_dir` read-only and points `extra_models_dir` at it, so everything already on disk appears as an `extra.*` model at **zero additional storage**. `lemonade_aliases` then binds a usable name to the auto-generated one.

`Qwen3.8-27B-GGUF` is the deliberate exception, downloaded even though the same GGUF is already mounted: the catalog entry declares the `mmproj` sidecar and the `mtp` label, and an auto-discovered bare GGUF carries neither. ~17 GB buys vision plus roughly double the decode.

### DwarfStar (`ds4`) was deployed, measured, and removed

[ds4](https://github.com/antirez/ds4) is antirez's self-contained DeepSeek-V4 engine, and Lemonade 11.8.0 ships it as an experimental backend. It was deployed here in full and then removed on the numbers, not on taste:

| DeepSeek-V4-Flash path | Decode | Quant | Residency | Telemetry |
|---|---:|---|---|---|
| Lemonade + `ds4` | 12.8 | ~2.3 bpw IQ2XXS | SSD-streamed, cannot be disabled | broken (`ttft=0`, `tps=0`) |
| **Lemonade + `llamacpp:rocm`** | **17.1** | **2.90 bpw**, verified 90.8% token-identical | **fully resident (103.8 GB)** | works |

Lemonade's llama.cpp build carries the `deepseek4` arch and is *newer* than the b10217 the `deepseek-v4` profile pins, so the second engine buys nothing. The 81 GB IQ2XXS download was deleted. See [PERFORMANCE.md](ansible_collections/nerdsrun/strix_halo_vllm/docs/PERFORMANCE.md#lemonade-server-1180) for the full measurement, including why `--ssd-streaming` is the likely cause and cannot be worked around.

### Three things that will cost you a day

> **1. `enable_dgpu_gtt` is mandatory on Strix Halo, and the failure is silent.** Lemonade sizes a device's memory pool from the JSON key it enumerated the GPU under, and only `amd_igpu` gets `max(vram, GTT)`. This APU enumerates as `amd_gpu`, so the pool reads as `vram_gb` alone — **0.5 GB**, the BIOS carveout — ignoring the 124 GB of GTT the fleet actually runs on. Models are then filtered out of the catalog with no error anywhere, which looks exactly like a wrong model name. The role sets this by default.
>
> **2. Browsers get 403 while curl gets 200.** Lemonade validates the `Origin` header and permits only loopback and desktop schemes, so the web UI opened over the LAN loads the page and then fails every chat request. Non-browser clients send no `Origin` at all, so the endpoint looks healthy from the command line while the UI is dead. `lemonade_allowed_origins` derives the box's LAN origin.
>
> **3. There is no Fedora RPM for 11.8.0.** The release notes link to one; the asset is not attached and the URL 404s. v11.7.0 published 13 assets, v11.8.0 published 6 — the Linux packages were withdrawn after the configuration-data-loss report that opens those notes. This deployment uses the container image, which is published and fits this collection's rootless-Podman idiom anyway.

### One long generation blocks everything

`max_loaded_models` is **1**. A realistic UI prompt ("build me a single-page app") runs tens of thousands of tokens, which at these rates is *tens of minutes*, and every other client — including `lemonade load` — queues behind it and eventually times out. That reads exactly like a hung server and is not one. `podman logs --tail 5 lemonade-server` shows a live token counter; a rising count means it is working, and restarting throws the work away.

---

## Models

### llama.cpp (Vulkan) -- GGUF Models

The `llamacpp` mode downloads and serves GGUF-quantized models via the Vulkan backend. Models are selected via profiles (see Modes above). The Vulkan backend bypasses a [known ROCm/HIP hang](https://github.com/ROCm/ROCm/issues/6027) with Qwen 3.5 models on gfx1151.

### vLLM (ROCm) -- HuggingFace Models

The `service`/`toolbox` modes use vLLM with ROCm. Default prefetch models:

| Model | Type | Size |
|---|---|---|
| `btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit` | MoE, 3B active | ~15GB |
| `btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit` | MoE, 3B active | ~8GB |
| `Qwen/Qwen3-14B-AWQ` | Dense, 14B | ~7GB |

Add custom models via `model_prefetch_extra: ["org/model-name"]`.

---

## Open WebUI

Set `ui_enabled: true` or use `mise run ui:up` to deploy [Open WebUI](https://github.com/open-webui/open-webui) as a chat interface.

| Setting | Value |
|---|---|
| Open WebUI URL | `http://localhost:3000` |
| Backend API URL | Auto-detected from `strix_halo_mode` (vLLM :8000, llama.cpp :8080) |
| API Key | `local-dev-key` (default) |

Open WebUI auto-connects to whichever backend is deployed. The container-to-container URL is resolved automatically via `host.containers.internal`.

---

## Security Notes

- **API Key**: Default is `local-dev-key` -- change it for any network-accessible deployment
- **Network Binding**: vLLM binds to `0.0.0.0:8000`, llama.cpp to `0.0.0.0:8080`. Set `firewall_open_vllm_port` / `firewall_open_llamacpp_port` to open in firewalld
- **seccomp=unconfined**: Required for ROCm GPU access (vLLM mode). Not needed for Vulkan (llama.cpp mode)
- **Quadlet files**: Deployed with `mode: 0600` -- API keys and tokens are rendered in the unit file
- **Rootless Podman**: All containers run rootless under the invoking user

---

## Mise Tasks

| Task | Description |
|---|---|
| `mise run bootstrap` | Install Ansible toolchain |
| `mise run ssh-key` | Pull SSH key from 1Password |
| `mise run deploy:toolbox` | Deploy toolbox mode (vLLM/ROCm) |
| `mise run deploy:service` | Deploy service mode (vLLM/ROCm) |
| `mise run deploy:llamacpp` | Deploy llama.cpp/Vulkan (big profile) |
| `mise run deploy:llamacpp:coder` | Deploy llama.cpp coder profile |
| `mise run deploy:llamacpp:fast` | Deploy llama.cpp fast profile |
| `mise run deploy:llamacpp:nemotron` | Deploy llama.cpp nemotron profile |
| `mise run deploy:llamacpp:lightning` | Deploy llama.cpp lightning profile |
| `mise run deploy:llamacpp:super` | Deploy llama.cpp super profile |
| `mise run deploy:llamacpp:coder-next` | Deploy llama.cpp coder-next profile |
| `mise run deploy:llamacpp:minimax` | Deploy llama.cpp minimax profile |
| `mise run deploy:llamacpp:deepseek` | Deploy llama.cpp deepseek profile (gated — see note) |
| `mise run deploy:all` | Full deployment |
| `mise run verify` | Run verification checks |
| `mise run uninstall` | Remove all components |
| `mise run ui:up` | Start Open WebUI |
| `mise run ui:down` | Stop Open WebUI |
| `mise run logs:vllm` | Tail vLLM logs |
| `mise run logs:llamacpp` | Tail llama.cpp logs |
| `mise run logs:ui` | Tail Open WebUI logs |
| `mise run benchmark` | Run LLM performance benchmark |
| `mise run lint` | Run ansible-lint + yamllint |
| `mise run test` | Run Molecule tests |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/dev/kfd` missing | `sudo modprobe amdgpu` |
| Permission denied on GPU | `sudo usermod -aG video,render $USER` then re-login |
| gfx1151 not detected | Update kernel: `sudo dnf upgrade --refresh` |
| vLLM won't start | `mise run logs:vllm` for ROCm errors |
| llama.cpp won't start | `systemctl --user reset-failed llamacpp-server` then retry |
| Open WebUI 500 error | `podman restart open-webui` |
| Slow inference | See [Performance Guide](ansible_collections/nerdsrun/strix_halo_vllm/docs/PERFORMANCE.md) |

See [Troubleshooting](ansible_collections/nerdsrun/strix_halo_vllm/docs/TROUBLESHOOTING.md) for details.

---

## Documentation

- [Getting Started](ansible_collections/nerdsrun/strix_halo_vllm/docs/GETTING_STARTED.md) -- Full setup walkthrough
- [Performance Tuning](ansible_collections/nerdsrun/strix_halo_vllm/docs/PERFORMANCE.md) -- Maximize tok/s on Strix Halo
- [Troubleshooting](ansible_collections/nerdsrun/strix_halo_vllm/docs/TROUBLESHOOTING.md) -- Fix common issues
- [Variables Reference](ansible_collections/nerdsrun/strix_halo_vllm/docs/VARIABLES.md) -- All configuration options
- [Architecture](ansible_collections/nerdsrun/strix_halo_vllm/docs/ARCHITECTURE.md) -- Collection design

---

## License

MIT
