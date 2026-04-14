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
mise run deploy:llamacpp:super    # llama.cpp — Nemotron Super 120B profile
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
| `super` | Nemotron-3-Super-120B-A12B | ~84 GB | 12B | ~22 | Reasoning, planning, orchestration |

> **Note:** The `nemotron` and `super` profiles require llama.cpp build **≥8351** (fixes [ggml-org/llama.cpp#20570](https://github.com/ggml-org/llama.cpp/issues/20570) — mamba-base.cpp assertion crash). Both are hybrid Mamba-Transformer architectures. The `super` profile uses ~84 GB at Q3_K_XL — do not run alongside other models on 128 GB systems.

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
| `mise run deploy:llamacpp:super` | Deploy llama.cpp super profile |
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
