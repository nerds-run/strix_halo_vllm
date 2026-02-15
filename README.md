# AMD Strix Halo vLLM Automation

**nerdsrun.strix_halo_vllm** -- Ansible collection for deploying vLLM on AMD Ryzen AI Max "Strix Halo" (gfx1151) APUs

![CI](https://img.shields.io/github/actions/workflow/status/nerdsrun/amdllmv/ci.yml?branch=main&label=CI)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

Enterprise-quality, idempotent Ansible automation that takes a Fedora system with AMD Strix Halo hardware from zero to a fully operational vLLM inference server. Supports toolbox mode for interactive development, service mode for persistent API endpoints, model weight prefetching, and an optional Open WebUI chat frontend.

**Upstream source of truth:** [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes/)

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
mise run deploy:toolbox   # interactive toolbox
mise run deploy:service   # systemd-managed vLLM API server
mise run deploy:all       # both modes

# 5. Verify
mise run verify

# 6. (Optional) Chat UI
mise run ui:up            # Open WebUI at http://localhost:3000
```

---

## Modes

| Mode | Variable | Description |
|---|---|---|
| `toolbox` | `strix_halo_mode: "toolbox"` | Interactive toolbox container for development |
| `service` | `strix_halo_mode: "service"` | Persistent vLLM server via systemd Quadlet |
| `both` | `strix_halo_mode: "both"` | Deploy both simultaneously |

---

## Model Prefetching

Default models (downloaded automatically):

| Model | Type | Size | tok/s (128GB system) |
|---|---|---|---|
| `btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit` | MoE, 3B active | ~8GB | ~35 |
| `btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit` | MoE, 3B active | ~15GB | ~25 |
| `dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16` | MoE, 3B active | ~46GB | ~18 |
| `Qwen/Qwen3-14B-AWQ` | Dense, 14B | ~7GB | ~12 |

MoE (Mixture of Experts) models are strongly recommended for Strix Halo -- they activate only a fraction of their parameters per token, giving much better speed on bandwidth-limited iGPUs. See the [Performance Guide](ansible_collections/nerdsrun/strix_halo_vllm/docs/PERFORMANCE.md).

Add custom models via `model_prefetch_extra: ["org/model-name"]`.

---

## Open WebUI

Set `ui_enabled: true` or use `mise run ui:up` to deploy [Open WebUI](https://github.com/open-webui/open-webui) as a chat interface.

| Setting | Value |
|---|---|
| Open WebUI URL | `http://localhost:3000` |
| vLLM API URL | `http://localhost:8000/v1` |
| API Key | `vllm_api_key_value` (default: `local-dev-key`) |

Open WebUI auto-connects to vLLM. The container-to-container URL is resolved automatically via `host.containers.internal`.

---

## Security Notes

- **API Key**: Default is `local-dev-key` -- change it for any network-accessible deployment
- **Network Binding**: vLLM binds to `0.0.0.0:8000` by default. Set `firewall_open_vllm_port: true` to open the port in firewalld (works in both toolbox and service modes), or restrict binding
- **seccomp=unconfined**: Required for ROCm GPU access inside containers. Does not disable other security mechanisms
- **Rootless Podman**: All containers run rootless under the invoking user

---

## Mise Tasks

| Task | Description |
|---|---|
| `mise run bootstrap` | Install Ansible toolchain |
| `mise run ssh-key` | Pull SSH key from 1Password |
| `mise run deploy:toolbox` | Deploy toolbox mode |
| `mise run deploy:service` | Deploy service mode |
| `mise run deploy:all` | Full deployment |
| `mise run verify` | Run verification checks |
| `mise run uninstall` | Remove all components |
| `mise run ui:up` | Start Open WebUI |
| `mise run ui:down` | Stop Open WebUI |
| `mise run logs:vllm` | Tail vLLM logs |
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
