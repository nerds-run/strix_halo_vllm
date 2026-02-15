# Getting Started

Step-by-step guide to deploy vLLM on AMD Strix Halo hardware.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Fedora 43+ |
| **Hardware** | AMD Ryzen AI Max "Strix Halo" APU (gfx1151) |
| **RAM** | 64 GB minimum, 128 GB recommended |
| **Disk** | 50 GB minimum; 200 GB+ for the full model set |
| **Devices** | `/dev/kfd` and `/dev/dri` present |
| **Task Runner** | [mise](https://mise.jdx.dev/) |

### Verify Hardware

```bash
# Confirm gfx1151 is detected
cat /sys/class/kfd/kfd/topology/nodes/*/properties | grep gfx_target_version
# Expected: gfx_target_version 110501

# Confirm device nodes exist
ls -la /dev/kfd /dev/dri/render*
```

If `/dev/kfd` is missing, load the driver:

```bash
sudo modprobe amdgpu
```

---

## Step 1 -- Install mise

```bash
curl https://mise.jdx.dev/install.sh | sh
echo 'eval "$(~/.local/bin/mise activate zsh)"' >> ~/.zshrc
source ~/.zshrc
```

See [mise docs](https://mise.jdx.dev/getting-started.html) for other shells or package managers.

---

## Step 2 -- Clone the Repository

```bash
git clone https://github.com/nerdsrun/amdllmv.git
cd amdllmv
```

---

## Step 3 -- Bootstrap

Install Ansible, linters, and Molecule:

```bash
mise run bootstrap
```

This installs the full toolchain into a Python virtual environment.

---

## Step 4 -- Configure Inventory

Copy the example inventory and fill in your target host details:

```bash
cp inventory/hosts.yml.example inventory/hosts.yml
$EDITOR inventory/hosts.yml
```

Set `ansible_host` to the IP/hostname of your Strix Halo machine and `ansible_user` to your SSH user.

### SSH Key Setup

The project stores the deploy key at `.ssh/framework_fedora` (gitignored). You have three options:

**Option A -- 1Password (default)**

The included `mise run ssh-key` task pulls a key from 1Password using the `op` CLI. By default it fetches the item `framework_fedora` from the `Infrastructure` vault. To use your own 1Password item, edit the `[tasks.ssh-key]` section in `mise.toml`:

```toml
# mise.toml – change these values to match your 1Password setup
op item get YOUR_ITEM_NAME \
  --vault YOUR_VAULT_NAME \
  --fields "private key" \
  --reveal
```

Then run:

```bash
mise run ssh-key
```

**Option B -- Manual key**

Place any SSH private key at `.ssh/framework_fedora`, or change `ansible_ssh_private_key_file` in your `inventory/hosts.yml` to point at your own key:

```bash
cp ~/.ssh/id_ed25519 .ssh/framework_fedora
chmod 600 .ssh/framework_fedora
```

**Option C -- SSH agent**

If your key is already loaded in an SSH agent, remove the `ansible_ssh_private_key_file` line from `inventory/hosts.yml` and the `SSH_AUTH_SOCK=` prefix from deploy tasks in `mise.toml`.

### Verify Connectivity

```bash
ansible -i inventory -m ping all
```

---

## Step 5 -- Configure Variables

Edit the group variables:

```bash
$EDITOR inventory/group_vars/all.yml
```

### Minimal Configuration

```yaml
---
strix_halo_mode: "toolbox"     # or "service" or "both"
```

### Recommended Configuration

```yaml
---
strix_halo_mode: "toolbox"
strix_halo_kernel_args_enabled: true
strix_halo_kernel_reboot_allowed: true
strix_halo_toolbox_update: true

# Open firewall ports for remote access
firewall_open_vllm_port: true

# Enable Open WebUI chat frontend
ui_enabled: true
firewall_open_ui_port: true
```

### Common Options

| Variable | Description |
|---|---|
| `strix_halo_mode` | `toolbox` (interactive), `service` (systemd), or `both` |
| `strix_halo_kernel_args_enabled` | Tune kernel for GPU memory (recommended) |
| `ui_enabled` | Deploy Open WebUI chat frontend |
| `vllm_primary_model` | Model to serve in service mode |
| `vllm_api_key_value` | API key (change from default for network access) |

See [Variables Reference](VARIABLES.md) for all options.

---

## Step 6 -- Deploy

Choose your deployment mode:

```bash
# Interactive toolbox (development / experimentation)
mise run deploy:toolbox

# Persistent vLLM API server (production)
mise run deploy:service

# Both modes
mise run deploy:all
```

### What the Playbook Does

1. **host_prereqs** -- Verifies hardware, groups, devices; installs podman/toolbox
2. **kernel_tuning** -- Adds GPU memory kernel parameters (if enabled); reboots if needed
3. **toolbox_mode** -- Pulls the vLLM image and creates a toolbox container; opens firewall port if enabled
4. **model_cache** -- Downloads model weights via aria2 or huggingface-cli
5. **podman_service** -- Creates a systemd Quadlet for the vLLM server (service mode)
6. **openwebui_ui** -- Deploys Open WebUI container (if enabled)
7. **verify** -- Runs post-deployment health checks

---

## Step 7 -- Verify

```bash
mise run verify
```

Or test manually:

```bash
# Toolbox mode -- enter the toolbox
toolbox enter vllm
rocm-smi                        # should show gfx1151
python -c "import vllm; print(vllm.__version__)"

# Service mode -- query the API
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer local-dev-key"

# Open WebUI
# Browse to http://<host>:3000
```

---

## Step 8 -- Start Using vLLM

### Toolbox Mode

Enter the toolbox and launch any of the prefetched models:

```bash
toolbox enter vllm
```

**Qwen3-Coder-30B MoE (4-bit) -- Fastest, ~35 tok/s:**
```bash
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit \
  --enforce-eager \
  --api-key local-dev-key
```

**Qwen3-Coder-30B MoE (8-bit) -- Balanced, ~25 tok/s:**
```bash
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit \
  --enforce-eager \
  --api-key local-dev-key
```

**Qwen3-Next-80B MoE (4-bit) -- Best quality, ~18 tok/s:**
```bash
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16 \
  --enforce-eager \
  --api-key local-dev-key
```

**Qwen3-14B Dense (AWQ) -- Smallest, ~12 tok/s:**
```bash
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve Qwen/Qwen3-14B-AWQ \
  --enforce-eager \
  --api-key local-dev-key
```

> The first request after launch will be slower as TunableOp benchmarks kernel variants. Subsequent requests use cached results.

### Service Mode

The vLLM server starts automatically via systemd:

```bash
# Check status
systemctl --user status vllm-server

# View logs
mise run logs:vllm

# Restart
systemctl --user restart vllm-server
```

### Open WebUI

Browse to `http://<host>:3000`. Select a model from the dropdown and start chatting. If models don't appear, ensure vLLM is running and try refreshing or restarting the Open WebUI container:

```bash
podman restart open-webui
```

---

## Next Steps

- [Performance Tuning](PERFORMANCE.md) -- Maximize tok/s on Strix Halo
- [Variables Reference](VARIABLES.md) -- All configuration options
- [Troubleshooting](TROUBLESHOOTING.md) -- Fix common issues
- [Architecture](ARCHITECTURE.md) -- Collection design and role structure
