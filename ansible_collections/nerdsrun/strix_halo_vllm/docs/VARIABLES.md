# Variables Reference

## Core Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `strix_halo_mode` | string | `"toolbox"` | Deployment mode: `toolbox`, `service`, `both`, or `llamacpp` |
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
| `strix_halo_kernel_args` | string | `"amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856"` | Kernel args to add (`amd_iommu=off` measured +6% memory bandwidth on gfx1151) |
| `strix_halo_kernel_args_remove` | list | `["iommu=pt"]` | Kernel args to remove via `grubby --remove-args` (needed when swapping superseded values like `iommu=pt` → `amd_iommu=off`) |
| `strix_halo_kernel_reboot_allowed` | bool | `false` | Allow automatic reboot after kernel arg changes |

## GPU Tuning Variables

Pin the amdgpu (gfx1151) iGPU to a high-performance state so Vulkan compute workloads don't get throttled by the default DVFS governor. Immediate effect via sysfs plus a systemd oneshot for boot persistence.

| Variable | Type | Default | Description |
|---|---|---|---|
| `gpu_tuning_enabled` | bool | `true` | Write sysfs and install systemd oneshot |
| `gpu_tuning_perf_level` | string | `"high"` | `power_dpm_force_performance_level` value: `auto`, `high`, `low`, or `manual` |
| `gpu_tuning_profile_mode` | int | `-1` | `pp_power_profile_mode` value. `-1` skips (gfx1151 doesn't expose this knob); `5` = COMPUTE on cards that do |
| `gpu_tuning_unit_name` | string | `"amdgpu-performance.service"` | systemd unit name for boot-time persistence |

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

## llama.cpp Server Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `llamacpp_image` | string | `"docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv"` | Container image (Vulkan backend) |
| `llamacpp_enabled` | bool | `true` | Enable the llama.cpp server |
| `llamacpp_host` | string | `"0.0.0.0"` | llama.cpp bind address |
| `llamacpp_port` | int | `8080` | llama.cpp listen port |
| `llamacpp_model_profile` | string | `"big"` | Model profile: `big`, `coder`, `fast`, `qwen38`, `qwen38-fp4`, `nemotron`, `lightning`, `super`, `coder-next`, `minimax`, `deepseek-v4`, or `deepseek` (gated -- see `unsupported_reason`) |
| `llamacpp_model_profiles` | dict | (see defaults) | Profile definitions. Per-profile keys: `repo`, `file`, `include`, `ctx_size`, `temp`, `top_p`, `top_k`, `min_p`, `jinja`, `repeat_penalty`, `presence_penalty`, `batch_size`, `ubatch_size`, `cache_type_k`, `cache_type_v`, `parallel_slots`, `mmproj_file`, `draft_file`, `draft_ngl`, `extra_args`, `unsupported_reason`. `repeat_penalty` / `presence_penalty` of `0` omit the flag entirely rather than passing `0`. `include` accepts a string (one glob) **or a list** — each pattern becomes its own repeated `--include` flag, which is required: `hf download --include "a" "b"` silently reparses the trailing patterns as positional filenames and discards `--include` altogether. `mmproj_file` renders `--mmproj` (vision projector; vision is silently off without it). `draft_file` renders `--model-draft` plus `--spec-type` (`spec_type`, default `draft-mtp`), `--spec-draft-ngl` (`draft_ngl`, default 99) and `--spec-draft-n-max` (`spec_n_max`, default 4); `draft_repo` points the drafter at a different HF repo from the target and triggers a second download; `draft_device` / `main_device` render `--spec-draft-device` / `-dev`. `chat_template_kwargs` (dict) renders `--chat-template-kwargs` as JSON — used to pin a default `reasoning_effort`.

**Backend is a per-profile property.** Most profiles use the global Vulkan `llamacpp_image`; two override it:

| Profile | Image | Build | Why |
|---|---|---|---|
| `qwen38-fp4` | `rocm-7.2.4-rocmfpx` | 211 | its quant uses ROCmFP4 tensor types that stock llama.cpp rejects |
| `deepseek-v4` | `rocm-7.2.4` (mainline) | 10217 | `deepseek4` has no Vulkan implementation; the fork costs 40% decode on this model |

Both set `devices: ['/dev/dri', '/dev/kfd']` and `podman_args: ['--ipc=host']`, which ROCm requires. Flag availability differs per build — see [TROUBLESHOOTING](TROUBLESHOOTING.md#llama-server-refuses-to-start-invalid-argument). A profile may also override host-level settings: `image` (per-profile container image — used by `qwen38-fp4` to select the ROCm/ROCmFPX image instead of the default Vulkan one), `devices` (list, default `['/dev/dri']`; ROCm needs `/dev/kfd` too), `env` (dict, merged over the global `llamacpp_env`), `podman_args` (list, rendered as a Quadlet `PodmanArgs=` line — `--ipc=host` is mandatory for ROCm), `no_mmap` (overrides the global `llamacpp_no_mmap`), and `backend` (label shown in the deploy summary) |
| `llamacpp_model_dir` | string | `"~/models"` | Directory for GGUF model storage |
| `llamacpp_ngl` | int | `999` | GPU layers to offload (999 = all) |
| `llamacpp_flash_attn` | bool | `true` | Enable flash attention |
| `llamacpp_no_mmap` | bool | `true` | Disable mmap (required for Strix Halo stability) |
| `llamacpp_batch_size` | int | `512` | Logical batch size for prompt processing (per-profile `batch_size` overrides) |
| `llamacpp_ubatch_size` | int | `0` | Physical (micro) batch size. `0` lets llama-server pick (default 512). **`2048` is the measured gfx1151 prefill peak** (311 t/s vs 169 at 1024); 4096 regresses. Also sets the compute-buffer size (~14 GiB at 2048), which is the main non-weight memory cost — see [PERFORMANCE.md](PERFORMANCE.md#vulkan-specific-tuning) |
| `llamacpp_thinking_enabled` | bool | `true` | Enable thinking/reasoning mode |
| `llamacpp_log_disable` | bool | `true` | Pass `--log-disable` to llama-server (reduces log noise). Set `false` to stream per-request `prompt eval time` / `eval time` to the journal |
| `llamacpp_cache_type_k` | string | `""` | KV cache key quantization (e.g. `q8_0`, `q4_0`) — per-profile `cache_type_k` overrides |
| `llamacpp_cache_type_v` | string | `""` | KV cache value quantization — per-profile `cache_type_v` overrides |
| `llamacpp_env` | dict | `{}` | Extra container environment vars rendered as `Environment=KEY=VAL` in the Quadlet. Useful for Vulkan/driver knobs (e.g. `RADV_PERFTEST`) |
| `llamacpp_extra_args` | list | `[]` | Additional llama-server CLI arguments (appended after any per-profile `extra_args`) |
| `llamacpp_api_key_enabled` | bool | `false` | Require API key authentication |
| `llamacpp_api_key_value` | string | `"local-dev-key"` | API key value |
| `llamacpp_container_name` | string | `"llamacpp-server"` | Container/service name |

## Lemonade Server Variables

Lemonade is a multi-model **router**, not a single-model server: one Quadlet unit loads and evicts models on demand across several engines. Deployed with `mise run deploy:lemonade` (playbook `lemonade.yml`), independent of `strix_halo_mode`.

| Variable | Type | Default | Description |
|---|---|---|---|
| `lemonade_enabled` | bool | `true` | Deploy and start the service |
| `lemonade_version` | string | `"11.8.0"` | Upstream version, used only in unit descriptions and output — the image digest is what actually pins it |
| `lemonade_image` | string | `ghcr.io/lemonade-sdk/lemonade-server@sha256:12a81cc2...` | Container image, **pinned by digest**. Both `latest` and `vX.Y.Z` move. There is no rocm/vulkan image variant: backend binaries are fetched at run time, so the image is GPU-agnostic |
| `lemonade_container_name` | string | `"lemonade-server"` | Quadlet unit and container name |
| `lemonade_host` | string | `"0.0.0.0"` | Bind address passed to `lemond`. Must be `0.0.0.0` for `PublishPort` to reach it. Note that in 11.8.0 `--host` became an *ephemeral* override that no longer persists to `config.json` |
| `lemonade_port` | int | `13305` | Host port published to the container's 13305 |
| `lemonade_api_key` | string | `""` | Bearer token for the OpenAI-compatible API, rendered as `LEMONADE_API_KEY`. Empty disables auth |
| `lemonade_devices` | list | `['/dev/kfd', '/dev/dri']` | ROCm needs the KFD compute device, not just the render node |
| `lemonade_supplementary_groups` | list | `['render', 'video']` | Host groups whose GIDs are resolved at run time and rendered as `GroupAdd=`. Rootless Podman only maps supplementary groups named explicitly, and the GIDs are not stable across hosts |
| `lemonade_podman_args` | list | `['--ipc=host']` | `--ipc=host` is **mandatory** for ROCm under rootless Podman — measured necessary *and* sufficient; `seccomp=unconfined` does not substitute. Without it the HSA runtime cannot map its shared-memory segments and model load dies with a misleading "Memory in use" |
| `lemonade_volumes` | dict | 4 named volumes | Volume name → container path. Models land in the Podman volume store in HuggingFace cache layout, **not** in `llamacpp_model_dir` — the two stacks cannot share a model file |
| `lemonade_backends` | list | `['llamacpp:rocm']` | Installed with `lemonade backends install`. Pulls llama.cpp b10470 plus the TheRock ROCm 7.14.0 gfx1151 runtime (~1.8 GB). `ds4:rocm` was deployed, measured and removed — see [PERFORMANCE](PERFORMANCE.md#lemonade-server-1180) |
| `lemonade_config` | dict | `{enable_dgpu_gtt: true, llamacpp.backend: rocm, extra_models_dir: /models}` | Applied with `lemonade config set k=v`. Dotted keys map to nested JSON. **`enable_dgpu_gtt` is mandatory here** — see below |
| `lemonade_models` | list | `['Qwen3.8-27B-GGUF']` | Models Lemonade **downloads**. Only put a model here when its catalog entry adds something the raw GGUF lacks — Qwen3.8-27B's declares the `mmproj` sidecar (vision) and the `mtp` label (speculative decoding), which is worth ~17 GB for vision plus roughly double the decode. Anything already under `llamacpp_model_dir` belongs in `lemonade_required_models` and costs nothing |
| `lemonade_required_models` | list | the Kevletesteur DeepSeek-V4 quant | Models that must resolve in the catalog but are **not** downloaded — they arrive via `extra_models_dir`. Deploy fails if one is missing, which distinguishes a broken mount from a wrong name |
| `lemonade_aliases` | dict | `{deepseek-v4: Kevletesteur_...}` | `alias: target` pairs bound with `lemonade alias add`. Auto-discovered `extra.*` names are derived from the directory, so they carry the uploader's handle and are unusable in a chat UI |
| `lemonade_extra_models_host_dir` | string | `{{ llamacpp_model_dir }}` | Host GGUF tree bind-mounted **read-only** so both stacks share one copy. Empty string disables the mount |
| `lemonade_extra_models_mount` | string | `"/models"` | Mount point inside the container; `extra_models_dir` points here |
| `lemonade_allowed_origins` | list | the box's own LAN origin | Browser origins allowed to call the API, rendered as `LEMONADE_ALLOWED_ORIGINS`. Loopback and desktop schemes are always permitted; **every other origin gets a 403 on POST while `GET /` still returns 200**, so the web UI loads over the LAN and then fails every chat request. curl is unaffected — it sends no `Origin` — which is why the endpoint looks healthy from the command line |
| `lemonade_stop_conflicting_services` | list | `['llamacpp-server']` | User units stopped (and `reset-failed`) before Lemonade takes the GPU. systemd cannot express `Conflicts=` here, so the guard lives in the play |
| `lemonade_gtt_drain_timeout` | int | `120` | Seconds to poll `mem_info_gtt_used` after stopping a conflicting unit. The driver frees GTT when the process exits, not when systemd returns |
| `firewall_open_lemonade_port` | bool | `false` | Open `lemonade_port/tcp` in firewalld |


### `extra_models_dir` — sharing the GGUF tree instead of copying it

Lemonade keeps downloads in HuggingFace cache layout and cannot read the flat `--local-dir` tree `llamacpp_service` builds, so without this every shared model is stored twice. The role bind-mounts `llamacpp_model_dir` read-only and points `extra_models_dir` at the mount, and everything already on disk becomes an `extra.*` model at no storage cost.

This is what lets `deepseek-v4` be served by Lemonade at **17.08 tok/s with 0 GB added** — the same 2.90 bpw quant, and the same throughput as the standalone profile.

Two practical notes:

- **Discovered names come from the directory**, e.g. `Kevletesteur_DeepSeek-V4-Flash-0731-StrixHalo-Verified-GGUF`. Bind something usable with `lemonade_aliases`.
- **Precedence is registered > imported > built-in**, so a downloaded model of the same bare name shadows a mounted one. `extra.NAME` addresses the mounted copy explicitly.

### `enable_dgpu_gtt` — why a model silently disappears without it

Lemonade decides whether a model fits by reading the device's memory pool, and it picks the sizing rule from the JSON key it enumerated the GPU under: only `amd_igpu` gets `MemoryAllocBehavior::Largest` (`max(vram_gb, virtual_mem_gb)`). Strix Halo enumerates as **`amd_gpu`**, so it falls through to `::Hardware`, which returns `vram_gb` alone — **0.5 GB**, the BIOS carveout — while the 124 GB of GTT this fleet actually runs on is `virtual_mem_gb` and is ignored.

Any model whose resident working set exceeds 0.5 GB is then filtered out. `enable_dgpu_gtt: true` switches the calculation to `::Unified` (`vram + GTT`) and the model reappears.

Two things make this hard to diagnose:

- **There is no error.** The model is simply absent from `lemonade list` and from `GET /api/v1/models`. `lemonade pull` against the name then fails, which reads as a typo.
- **Only *streaming* backends are affected.** The check applies to backends that read the model from disk on demand — `ds4` — using `min_resident_gb` as the working set. Every llama.cpp model stays visible either way, so the symptom presents as a ds4-specific problem rather than a device-detection one.

`disable_model_filtering: true` also makes it appear, but by turning off every size check; `enable_dgpu_gtt` fixes the arithmetic instead and is the right key.

## Firewall Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `firewall_open_vllm_port` | bool | `false` | Open vLLM port (8000) in firewalld |
| `firewall_open_llamacpp_port` | bool | `false` | Open llama.cpp port (8080) in firewalld |
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
| `openwebui_openai_api_base_url` | string | `""` (auto-resolved) | Backend API URL (auto-detected from `strix_halo_mode`) |
| `openwebui_openai_api_key` | string | `"local-dev-key"` | API key for backend connection |
| `openwebui_container_name` | string | `"open-webui"` | Container name |

## Verify Variables

The `verify` role is tagged `never` and runs on demand only (`mise run verify`).

| Variable | Type | Default | Description |
|---|---|---|---|
| `verify_log_lines` | int | `50` | Journal lines to pull per service when reporting deployment status |

## Uninstall Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `strix_halo_uninstall_purge_cache` | bool | `false` | Also remove cached data and volumes |
