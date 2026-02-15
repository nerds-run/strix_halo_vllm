# Variables Reference

## Core Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `strix_halo_mode` | string | `"toolbox"` | Deployment mode: `toolbox`, `service`, or `both` |
| `strix_halo_image` | string | `"docker.io/kyuz0/vllm-therock-gfx1151:latest"` | Container image for vLLM |
| `strix_halo_toolbox_name` | string | `"vllm"` | Name of the toolbox container |
| `strix_halo_target_user` | string | `"{{ ansible_user_id }}"` | User for rootless podman operations |
| `strix_halo_require_devices` | bool | `true` | Fail if GPU device nodes are missing |
| `strix_halo_devices` | list | `[/dev/dri, /dev/kfd]` | Required device nodes |
| `strix_halo_groups` | list | `[video, render]` | Groups the target user must belong to |
| `strix_halo_security_opt_seccomp_unconfined` | bool | `true` | Run containers with seccomp=unconfined |
| `strix_halo_toolbox_update` | bool | `false` | Force recreation of toolbox with latest image |

## Kernel Tuning Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `strix_halo_kernel_args_enabled` | bool | `false` | Enable kernel parameter tuning |
| `strix_halo_kernel_args` | string | `"iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856"` | Kernel args to add |
| `strix_halo_kernel_reboot_allowed` | bool | `false` | Allow automatic reboot after kernel arg changes |

## RDMA Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `strix_halo_rdma_enabled` | bool | `false` | Enable RDMA/InfiniBand device passthrough |
| `strix_halo_rdma_device_path` | string | `"/dev/infiniband"` | Path to RDMA device |

## Model Prefetch Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `model_prefetch_enabled` | bool | `true` | Enable model prefetching |
| `model_prefetch_strict` | bool | `false` | Fail on download errors (false = warn and continue) |
| `model_prefetch_list_default` | list | `[btbtyler09/Qwen3-..., Qwen/Qwen3-14B-AWQ]` | Default models to download |
| `model_prefetch_extra` | list | `[]` | Additional models to download |
| `model_prefetch_include_kimi_k25` | bool | `false` | Include Kimi-K2.5 (~400GB+) |
| `model_prefetch_kimi_id` | string | `"moonshotai/Kimi-K2.5"` | Kimi model identifier |
| `huggingface_cache_dir` | string | `"~/.cache/huggingface"` | HuggingFace cache directory |
| `vllm_cache_dir` | string | `"~/.cache/vllm"` | vLLM cache directory |
| `hf_token` | string | `""` | HuggingFace API token for gated models |
| `download_accelerator` | string | `"aria2"` | Download tool (aria2 or huggingface-cli) |
| `model_prefetch_min_free_gb` | int | `200` | Minimum free disk space warning threshold |

## vLLM Server Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `vllm_enabled` | bool | `true` (in service role) | Enable the vLLM server |
| `vllm_host` | string | `"0.0.0.0"` | vLLM bind address |
| `vllm_port` | int | `8000` | vLLM listen port |
| `vllm_primary_model` | string | `"btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit"` | Model to serve |
| `vllm_api_key_enabled` | bool | `true` | Require API key authentication |
| `vllm_api_key_value` | string | `"local-dev-key"` | API key value |
| `vllm_extra_args` | list | `[]` | Additional vLLM CLI arguments |
| `vllm_container_name` | string | `"vllm-server"` | Container/service name |

## Firewall Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `firewall_open_vllm_port` | bool | `false` | Open vLLM port in firewalld (applies in both toolbox and service modes) |
| `firewall_open_ui_port` | bool | `false` | Open WebUI port in firewalld |

## Open WebUI Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `ui_enabled` | bool | `false` | Enable Open WebUI deployment |
| `ui_type` | string | `"openwebui"` | UI type (only openwebui supported) |
| `openwebui_image` | string | `"ghcr.io/open-webui/open-webui:main"` | Open WebUI container image |
| `openwebui_port` | int | `3000` | Open WebUI listen port |
| `openwebui_data_volume` | string | `"open-webui"` | Podman volume for persistent data |
| `openwebui_auth_enabled` | bool | `true` | Enable authentication (false shows warning) |
| `openwebui_openai_api_base_url` | string | `"http://127.0.0.1:8000/v1"` | vLLM API base URL |
| `openwebui_openai_api_key` | string | `vllm_api_key_value` | API key for vLLM connection |
| `openwebui_container_name` | string | `"open-webui"` | Container name |

## Uninstall Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `strix_halo_uninstall_purge_cache` | bool | `false` | Also remove cached data and volumes |
