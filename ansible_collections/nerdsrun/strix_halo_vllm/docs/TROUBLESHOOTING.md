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

## Open WebUI Can't Connect to vLLM

**Symptoms:** Open WebUI shows "connection refused", 500 errors, or no models available

**Causes:**
- vLLM not running or not yet ready
- Wrong API URL or key
- Container networking issue
- Container needs restart after changing vLLM model

**Solutions:**
```bash
# Verify vLLM is responding
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer local-dev-key"

# Check Open WebUI container env
podman inspect open-webui | grep -A2 OPENAI

# The container uses host.containers.internal to reach the host
# Verify this resolves correctly
podman exec open-webui getent hosts host.containers.internal

# Restart Open WebUI (required after changing vLLM model)
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

## Complete Reset

```bash
# Remove everything
mise run uninstall

# With cache purge
ansible-playbook ansible_collections/nerdsrun/strix_halo_vllm/playbooks/uninstall.yml \
  -i inventory -e strix_halo_uninstall_purge_cache=true
```
