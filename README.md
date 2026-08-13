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
mise run deploy:llamacpp:deepseek # llama.cpp — DeepSeek-V4-Flash (gated, see note)
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

### llama.cpp Model Profiles

When using `llamacpp` mode, select a model profile with `llamacpp_model_profile` or via mise:

| Profile | Model | Size | Active Params | tok/s | Use Case |
|---|---|---|---|---|---|
| `big` (default) | Qwen3.5-122B-A10B | 77 GB | 10B | ~22 | Reasoning, vision, general |
| `coder` | Qwen3-Coder-30B-A3B | ~20 GB | 3B | ~83 | Coding, tool-use, agentic |
| `fast` | Qwen3.5-35B-A3B | ~20 GB | 3B | ~59 | Fast general + vision |
| `nemotron` | Nemotron-3-Nano-30B-A3B | ~20 GB | 3B | ~95 | Coding, agentic, reasoning |
| `lightning` | NVIDIA-Nemotron-3.5-Lightning-30B-A3B | ~35 GB | 3B | **51.4** | Fast agentic executor — tool use, code review, high-volume delegation |
| `super` | NVIDIA-Nemotron-3-Super-120B-A12B | ~63 GB | 12B | **15.6** | Reasoning, planning, tool-calling (1M ctx native) |
| `coder-next` | Qwen3-Coder-Next-80B-A3B | ~85 GB | 3B | **42.3 fresh / 27.5 at 32K** | Agentic coding delegation worker (challenger to minimax) |
| `minimax` | MiniMax-M2.7 (229B MoE) | ~108 GB | 10B | **28.1 fresh / 17.9 at 32K** | Long-context agentic, tool-use, reasoning |
| `deepseek` ⚠️ | DeepSeek-V4-Flash-0731 (284B MoE) | ~104 GB | 13B | TBD | Max-capability agentic coding, long-context — **gated, see note** |

> **Note:** The `nemotron` and `super` profiles require llama.cpp build **≥8351** (fixes [ggml-org/llama.cpp#20570](https://github.com/ggml-org/llama.cpp/issues/20570) — mamba-base.cpp assertion crash). Both are hybrid Mamba-Transformer architectures. The `super` profile runs at the **full native 1,048,576-token context** in ~73 GiB total on a 128 GB Strix Halo. The hybrid LatentMoE design means only 8 of 89 layers carry KV at all (the rest are constant-state Mamba-2 / MoE), so a 1M-token window costs just 2.25 GiB of KV cache — 1M is effectively free versus 512K. Non-weight memory is dominated by the ~14 GiB ubatch compute buffer instead, leaving ~50 GiB spare. Tool calling rides on `--jinja` + the GGUF's chat template; the vLLM-only `--tool-call-parser qwen3_coder` / `--reasoning-parser super_v3` flags are NOT supported by `llama-server` and will crash it.
>
> **Nemotron-3.5-Lightning (`lightning`):** Released 2026-08-11; the successor family to the `nemotron` (Nano) profile, same 30B-A3B size class. Hybrid Mamba-2 + MoE + Attention (GGUF arch `nemotron_h_moe`), up to 1M native context, [OpenMDW-1.1](https://openmdw.ai/license/1-1/) license. **Build requirement:** the arch is present in llama.cpp build **8985**, which the pinned `kyuz0/amd-strix-halo-toolboxes:vulkan-radv` image ships — verified by inspecting the image's `libllama.so`, so no gate beyond the existing ≥8351 note applies. Ships at **Q8_0 (35.0 GB)**: near-lossless, and at this size the 128 GB box has abundant headroom, so there is no reason to quantize harder. Context defaults to **262144**. **Measured on hardware:** at `-c 262144` with 2 slots the whole thing sits in **~38 GiB**, leaving ~87 GiB free — so 262K is conservative and there is ample room to raise it. Parallel slots default to **2** via the per-profile `parallel_slots` variable; confirmed live as `n_slots = 2, n_ctx_slot = 131072` — **slots subdivide `ctx_size`**, so two slots at 262K give each request a 131K window. Raise `ctx_size` alongside slots if you need the full window per request.
>
> At ~38 GiB it comfortably co-resides with `super` (~73 GiB) — but **not** with `coder-next` (~85 GiB) or `minimax` (~108 GiB), which would exceed 125 GiB together.
>
> **On the Q8 choice and speed:** measured decode is **51.4 tok/s**, against `nemotron`'s ~95 — but that gap is the quant, not the model. Decode here is memory-bandwidth-bound, and Q8_0 moves ~1.75× the bytes of `nemotron`'s Q4_K_XL, which predicts ~54 tok/s. The measurement lands right on that. **If you want the executor tier faster, drop this profile to a Q4/Q5 tier** and expect roughly nemotron-class decode; Q8 buys output fidelity, not throughput. Prefill, by contrast, is enormous — **1246 t/s at 1.6K and 1455 t/s at 6K prompts**, ~4× the `super` profile, because only 3B params are active per token.
>
> **Qwen3-Coder-Next (`coder-next`):** 80B total / 3B active, MoE with hybrid linear attention (GGUF arch `qwen3next`), 256K native context. Published as **`unsloth/Qwen3-Coder-Next-GGUF`** — there is no size suffix in the repo name. Ships at **Q8_0 (84.8 GB, 3 shards)**: near-lossless, chosen deliberately over IQ4 because this profile is a tool-calling delegation worker where the failure mode that matters is quant-induced malformed tool-call JSON, not a benchmark point. **Measured: 42.3 tok/s fresh, 40.9 at 8K — Q8 clears the 30 tok/s bar, so it stays.** Q6_K (65.6 GB) remains documented as a fallback but is not needed. **Note the context falloff:** decode drops to **27.5 tok/s at 32K (−35%)**, unlike the Nemotron profiles which stay flat — budget for roughly two-thirds throughput in long agent sessions. Prefill is 840 t/s at 6K. Resident footprint is **87 GiB** at the full 262144 context, which fits with ~38 GiB spare but does **not** co-reside with `lightning`. Context defaults to the **full native 262144** — hybrid linear attention keeps KV cheap and ~25 GB of post-weights headroom holds it. Parallel slots default to **2** (`parallel_slots`), tunable to 4 for multi-agent delegation. **This model is non-thinking** — its card states it never emits `<think>` blocks, so `--reasoning-format auto` is a no-op here and is kept only for consistency. **~85 GB resident + KV — do not run alongside other models.** This profile is a *challenger* to `minimax`, which remains the incumbent big coder until A/B'd on real tickets.
>
> **MiniMax-M2.7:** Ships at UD-IQ4_XS (~108 GB) — the tightest fit on 128 GB unified memory. Do not run alongside other models. Tool calling rides on `--jinja` + the chat template baked into the GGUF, with `--reasoning-format auto` to surface the model's reasoning blocks (llama.cpp does not support the vLLM-style `--tool-call-parser` / `--reasoning-parser` flags referenced in the MiniMax docs). Per [Unsloth docs](https://huggingface.co/unsloth/MiniMax-M2.7-GGUF), do **NOT** run these GGUFs on CUDA 13.2 (produces gibberish) — this deployment uses Vulkan, so no action needed.
>
> **DeepSeek-V4-Flash (`deepseek`) — ⚠️ EXPERIMENTAL, GATED, DOES NOT CURRENTLY RUN.** 284B total / 13B active, MoE with hybrid CSA/HCA sparse attention (GGUF arch `deepseek4`), architecture supports up to 1M context. The profile is defined in full and `mise run deploy:llamacpp:deepseek` resolves it end to end, but the deploy **stops at a guard task** rather than downloading ~104 GB and starting a server that cannot load. Two independent blockers:
>
> 1. **Vulkan op support is missing (probably).** DeepSeek-V4's attention ops (`LIGHTNING_INDEXER`, `DSV4_HC_PRE`/`COMB`/`POST`) appear upstream for **Metal and SYCL only** — a GitHub code search for `DSV4` under `ggml/src/ggml-vulkan` returns **0 hits** against **6** under `ggml/src/ggml-metal`. The architecture landed in [#24162](https://github.com/ggml-org/llama.cpp/pull/24162) (2026-06-29) and the Flash-0731 chat template in [#26398](https://github.com/ggml-org/llama.cpp/pull/26398) (2026-08-03).
>
>    **Updated for build 10400:** the toolbox image now *does* register the `deepseek4` architecture — build 8985 did not (verified by scanning `libllama.so` with `minimax-m2`/`qwen3moe` as passing controls). So the model may now load and then fail, or silently fall back to CPU, on the missing ops rather than being rejected at load. An attempt to check the op strings inside `libggml-vulkan.so` was **inconclusive** — the `FLASH_ATTN_EXT` control also came back absent despite flash attention demonstrably working, so those strings aren't reliably present in that binary. **Settling this requires actually attempting a ~104 GB load.** The gate stays closed by default until someone does; clear it with `-e llamacpp_allow_unsupported=true` to find out.
> 2. **The commonly-cited quant does not fit.** `UD-Q3_K_XL` is **128.2 GB** on Hugging Face — roughly 119 GiB against 125 GiB of total system RAM, before compute buffers or KV. It cannot load at any context length. The frequently-quoted "~103 GB" figure corresponds to **`UD-IQ3_XXS` (104.2 GB)**, which is the largest tier that actually fits and is what this profile uses. This substitution is deliberate and flagged rather than silent.
>
> Defaults are `-c 131072` with **KV cache quantization on by default** (`-ctk q8_0 -ctv q8_0`) per spec — both **unvalidated**, because the model cannot be loaded to validate them. **KV bytes-per-token is not reported for this architecture**: it is read from llama.cpp's load logs, and there are none. An estimate presented as a measurement would be worse than the gap. Once Vulkan support ships, measure it and raise `ctx_size` to what the headroom allows. Note also that **tool-call reliability is more fragile at 3-bit**; Unsloth's UD dynamic quants mitigate this by keeping attention layers at higher precision. To attempt the deploy anyway: `-e llamacpp_allow_unsupported=true`.

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
