# Troubleshooting

## /dev/dri or /dev/kfd Missing

**Symptoms:** Play fails with "Required device /dev/kfd does not exist"

**Causes:**
- `amdgpu` kernel driver not loaded
- Wrong kernel version (too old for Strix Halo)
- Running in a VM without GPU passthrough

**Solutions:**
```bash
# Check if amdgpu is loaded
lsmod | grep amdgpu

# Load it manually
sudo modprobe amdgpu

# Check dmesg for errors
dmesg | grep -i amdgpu

# Verify devices appeared
ls -la /dev/kfd /dev/dri/
```

## Permission Denied on GPU Devices

**Symptoms:** ROCm operations fail with permission errors inside containers

**Solutions:**
```bash
# Add user to required groups
sudo usermod -aG video,render $USER

# IMPORTANT: log out and back in for group changes to take effect
# Or use newgrp to activate in current shell
newgrp video
newgrp render

# Verify group membership
id -nG | grep -E 'video|render'

# Check device permissions
ls -la /dev/kfd /dev/dri/render*
```

## gfx1151 Not Detected

**Symptoms:** Play fails with "Strix Halo gfx1151 hardware not detected. Found: ..."

**Causes:**
- Not running on Strix Halo hardware
- Kernel too old for gfx1151 support
- amdgpu firmware missing

**Solutions:**
```bash
# Check what GPU is detected
cat /sys/class/kfd/kfd/topology/nodes/*/properties | grep gfx_target_version

# Update kernel and firmware
sudo dnf upgrade --refresh
sudo dnf install linux-firmware

# Reboot after kernel update
sudo reboot
```

## Toolbox Creation Failures

**Symptoms:** `toolbox create` fails

**Solutions:**
```bash
# Check if image was pulled successfully
podman images | grep vllm-therock-gfx1151

# Try pulling manually
podman pull docker.io/kyuz0/vllm-therock-gfx1151:latest

# Remove failed toolbox and retry
toolbox rm -f vllm
mise run deploy:toolbox

# Check podman storage
podman system info
```

## llama.cpp Service Won't Start

**Symptoms:** Service fails or exits immediately after deploy

**Solutions:**
```bash
# Check if systemd is in "restart too quickly" state
systemctl --user status llamacpp-server

# Reset failed state and restart
systemctl --user reset-failed llamacpp-server
systemctl --user daemon-reload
systemctl --user start llamacpp-server

# Check logs (--log-disable is on by default, check journal)
journalctl --user -u llamacpp-server --no-pager -n 50

# Verify model file exists at expected path
ls -lh ~/models/*/UD-Q4_K_XL/

# Check the Quadlet unit file
cat ~/.config/containers/systemd/llamacpp-server.container

# Test manually with logging enabled
podman run --rm --device /dev/dri -v ~/models:/models:Z \
  docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  llama-server -m /models/<repo_dir>/UD-Q4_K_XL/<model>.gguf \
  -ngl 999 -fa 1 --no-mmap --ctx-size 16384 --host 0.0.0.0 --port 8080
```

## Qwen 3.5 Hangs on ROCm

**Symptoms:** Qwen 3.5 models hang during tensor loading with ROCm/HIP backend

**Cause:** Known bug with Qwen 3.5 hybrid architecture (SSM/DeltaNet + MoE) on gfx1151. Tracked at [ROCm#6027](https://github.com/ROCm/ROCm/issues/6027).

**Solution:** Use the Vulkan backend instead (`strix_halo_mode: "llamacpp"`). This is what the `llamacpp_service` role does.

## vLLM Service Won't Start

**Symptoms:** Service fails to start or crashes immediately

**Solutions:**
```bash
# Check service status
systemctl --user status vllm-server

# View full logs
journalctl --user -u vllm-server --no-pager -n 100

# Or via podman
podman logs vllm-server

# Check if GPU is accessible inside the container
podman exec vllm-server rocm-smi

# Check if port is already in use
ss -tlnp | grep 8000

# Verify the Quadlet file
cat ~/.config/containers/systemd/vllm-server.container

# Restart after fixes
systemctl --user daemon-reload
systemctl --user restart vllm-server
```

## Model Download Failures

**Symptoms:** `model_cache` role fails during download

**Causes:**
- Network connectivity issues
- Missing HuggingFace token for gated models
- Insufficient disk space
- `huggingface-cli` renamed to `hf` in newer versions

**Solutions:**
```bash
# Check disk space
df -h ~/.cache/huggingface

# Test HuggingFace connectivity
huggingface-cli whoami
# If "command not found", try:
hf whoami

# Set token for gated models
# In inventory/group_vars/all.yml:
# hf_token: "hf_your_token_here"

# Retry with strict mode off (default)
# model_prefetch_strict: false  # warns but continues
```

## Open WebUI Can't Connect to Backend

**Symptoms:** Open WebUI shows "connection refused", 500 errors, or no models available

**Causes:**
- Backend not running or not yet ready
- Wrong API URL or key
- Container networking issue
- Container needs restart after switching models/backends

**Solutions:**
```bash
# Verify backend is responding (use correct port)
curl http://127.0.0.1:8080/v1/models    # llama.cpp
curl http://127.0.0.1:8000/v1/models    # vLLM

# IMPORTANT: Use 127.0.0.1, not localhost (IPv6 issue on Fedora)

# Check Open WebUI container env
podman inspect open-webui | grep -A2 OPENAI

# The container uses host.containers.internal to reach the host
podman exec open-webui curl -s http://host.containers.internal:8080/v1/models

# Restart Open WebUI (required after switching backends)
podman restart open-webui
```

## Open WebUI Health Check Fails (IPv6)

**Symptoms:** Ansible health check times out even though the container is running. Curl to `localhost` gives "Connection reset by peer".

**Cause:** On Fedora 43+, `localhost` resolves to `::1` (IPv6), but Podman's pasta network driver binds only on `0.0.0.0` (IPv4).

**Solutions:**
```bash
# Test IPv4 explicitly
curl http://127.0.0.1:3000

# If that works but localhost doesn't, the issue is IPv6
# The Ansible role already uses 127.0.0.1 for health checks
# For manual testing, always use 127.0.0.1 instead of localhost
```

## Slow Inference (Low tok/s)

**Symptoms:** Getting much lower tok/s than expected (e.g. 6 tok/s instead of 12+ on Qwen3-14B-AWQ)

**Causes:**
- TunableOp not enabled (most common)
- Using a dense model instead of MoE
- Missing kernel parameters
- HIP graph capture overhead

**Solutions:**

1. **Enable TunableOp** (biggest impact -- ~2x speedup):
```bash
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve <model> --enforce-eager --api-key local-dev-key
```

2. **Use MoE models** instead of dense models -- see [Performance Guide](PERFORMANCE.md)

3. **Verify kernel parameters** are applied:
```bash
cat /proc/cmdline | tr ' ' '\n' | grep -E 'iommu|gttsize|pages_limit'
# Should show: iommu=pt, amdgpu.gttsize=126976, ttm.pages_limit=32505856
```

4. **Use --enforce-eager** to disable HIP graph capture overhead

## XCCL Warning on Startup

**Symptoms:** vLLM logs show "XCCL is not enabled" or XCCL-related warnings

**Impact:** None. XCCL is for multi-GPU communication. Strix Halo has a single iGPU, so XCCL is irrelevant. The warning is harmless and can be ignored.

## SSH Connection Issues

**Symptoms:** Ansible can't reach the target host

**Solutions:**
```bash
# Test SSH manually
ssh user@target-host

# Check SSH key
ssh-add -l

# Pull key from 1Password (if configured)
mise run ssh-key

# Test Ansible connectivity
ansible -i inventory -m ping all

# Check inventory file
cat inventory/hosts.yml
```

## Kernel Parameter Issues

**Symptoms:** Unified memory not working optimally

**Solutions:**
```bash
# Check current kernel parameters
cat /proc/cmdline

# Verify grubby applied changes
grubby --info=ALL | grep args

# After adding kernel args, REBOOT is required
# The role won't auto-reboot unless strix_halo_kernel_reboot_allowed: true
sudo reboot
```

## SELinux Denials

**Symptoms:** Container operations fail with AVC denials

**Solutions:**
```bash
# Check for SELinux denials
sudo ausearch -m AVC -ts recent

# The :Z volume mount flag should handle most SELinux contexts
# If issues persist, check audit log
sudo sealert -a /var/log/audit/audit.log

# As a last resort (not recommended for production):
# sudo setenforce 0
```

## Collecting Logs

```bash
# llama.cpp service logs
mise run logs:llamacpp

# vLLM service logs
mise run logs:vllm

# Open WebUI logs
mise run logs:ui

# System journal for GPU issues
journalctl -k | grep -i amdgpu

# Full diagnostic bundle
podman ps -a
podman images
systemctl --user list-units | grep vllm
cat /proc/cmdline
rocm-smi 2>/dev/null || echo "rocm-smi not available on host"
```

## OOM Triage — llama.cpp Service Killed (exit 137)

**Symptoms:** `systemctl --user status llamacpp-server` shows `status=137/n/a` and "A process of this unit has been killed by the OOM killer" in the journal. Service auto-restarts (`NRestarts` increments) but there's a 30-60s outage while the model reloads.

**Find the actual cause in dmesg:**

```bash
# Look for the process that triggered OOM (not necessarily llama-server)
sudo dmesg -T --level=err,warn | grep -iE "invoked oom-killer|Killed process" | tail -5
```

The triggering process is often not `llama-server` itself — common culprits on Fedora include `sssd_kcm` (Kerberos cache manager), `podman cleanup`, or heavy log rotation. The kernel picks llama-server as the victim because it's the largest user process (`oom_score_adj=200`).

**Immediate recovery:**

```bash
# If systemd got stuck on restart backoff:
systemctl --user reset-failed llamacpp-server
systemctl --user restart llamacpp-server

# Watch model reload (can take 30-60s with --no-mmap + 100 GB weights):
journalctl --user -u llamacpp-server -f
```

**Root-cause fixes (in rough order of impact):**

| Cause | Fix |
|---|---|
| **Tight model + unbounded prompt cache** (e.g. MiniMax 108 GB + `--cache-reuse` at default 8 GB) | Add `--cache-ram 2048` to the profile's `extra_args` to cap the in-memory prompt cache pool. q8_0 KV doubles snapshot size vs q4_0 — prefer q4_0 on oversized models. |
| **Oversized `ctx_size`** | Drop to the smallest value that matches your working context. Summary-resetting around 40-50K with `ctx_size: 49152` gives ~5 GB headroom on 128 GB systems |
| **Other processes grabbing RAM mid-inference** | Disable unused services (e.g. `systemctl disable sssd_kcm` if you don't use Kerberos). Inspect `ps auxf --sort=-rss | head` to find transient allocators |
| **Cache-reuse snapshot write fails mid-allocation** | Journal will show `srv prompt_save: saving prompt with length N, total state size = X MiB` right before the OOM. Lower `--cache-ram` or drop `--cache-reuse` to eliminate the snapshot path |

**Verify free memory is comfortable:**

```bash
ansible -i inventory all -m shell -a 'free -h | head -2'
# Target: >5 GB available on 128 GB systems for MiniMax-class models
```

## llama-server Refuses to Start ("invalid argument")

**Symptoms:** Service fails within 1s of start, journal shows `error: invalid argument: --<flag>`.

**Cause:** A model card or recipe specified flags that are specific to another inference engine (vLLM / TGI). llama.cpp has its own CLI surface.

**Common offenders:**

| Model-card flag | llama.cpp equivalent |
|---|---|
| `--tool-call-parser minimax_m2` | `--jinja` + the GGUF's embedded chat template |
| `--reasoning-parser minimax_m2_append_think` | `--reasoning-format auto` |
| `--enforce-eager` | (vLLM-only; no equivalent needed on llama.cpp) |
| `--max-model-len N` | `--ctx-size N` |
| `--gpu-memory-utilization` | (vLLM-only) |

When in doubt: `podman run --rm <image> llama-server --help 2>&1 | grep -i <flag_stem>` to verify a flag actually exists in the build.

## GPU Perf Level Reverted After Reboot

**Symptoms:** `cat /sys/class/drm/card1/device/power_dpm_force_performance_level` returns `auto` after a reboot, decode tok/s is lower than expected.

**Cause:** The `amdgpu-performance.service` systemd unit (installed by the `gpu_tuning` role) isn't enabled or failed at boot.

**Fix:**

```bash
sudo systemctl status amdgpu-performance.service
# if disabled:
sudo systemctl enable --now amdgpu-performance.service
```

Re-run `mise run deploy:llamacpp` to reinstall/re-enable the unit idempotently.

## Complete Reset

```bash
# Remove everything
mise run uninstall

# With cache purge
ansible-playbook ansible_collections/nerdsrun/strix_halo_vllm/playbooks/uninstall.yml \
  -i inventory -e strix_halo_uninstall_purge_cache=true
```
