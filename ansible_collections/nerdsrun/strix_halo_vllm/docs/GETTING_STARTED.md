# Getting Started

Step-by-step guide to deploy LLM inference on AMD Strix Halo hardware. Supports two backends: **vLLM** (ROCm) and **llama.cpp** (Vulkan).

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
strix_halo_mode: "llamacpp"    # or "toolbox", "service", "both"
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
| `strix_halo_mode` | `toolbox`, `service`, `both` (vLLM/ROCm), or `llamacpp` (Vulkan) |
| `strix_halo_kernel_args_enabled` | Tune kernel for GPU memory (recommended) |
| `ui_enabled` | Deploy Open WebUI chat frontend |
| `vllm_primary_model` | Model to serve in service mode |
| `vllm_api_key_value` | API key (change from default for network access) |

See [Variables Reference](VARIABLES.md) for all options.

---

## Step 6 -- Deploy

Choose your deployment mode:

```bash
# --- llama.cpp (Vulkan) --- recommended for Qwen 3.5 models
mise run deploy:llamacpp          # 122B model (default, ~22 tok/s)
mise run deploy:llamacpp:coder    # Coder 30B (~83 tok/s)
mise run deploy:llamacpp:fast     # Fast 35B (~59 tok/s)
mise run deploy:llamacpp:nemotron # Nemotron Nano 30B (~95 tok/s, hybrid Mamba-Transformer)
mise run deploy:llamacpp:super    # Nemotron Super 120B (~22 tok/s, needs llama.cpp ≥8351, ~84 GB)
mise run deploy:llamacpp:minimax  # MiniMax-M2.7 229B/10B (~108 GB, long-context agentic + tool-use)

# --- vLLM (ROCm) ---
mise run deploy:toolbox           # Interactive toolbox
mise run deploy:service           # Persistent systemd service

# --- Full deployment ---
mise run deploy:all               # site.yml (mode-dependent)
```

### What the Playbook Does

1. **host_prereqs** -- Verifies hardware, groups, devices; installs podman/toolbox
2. **kernel_tuning** -- Adds GPU memory kernel parameters (if enabled); reboots if needed
3. **toolbox_mode** -- Pulls vLLM image and creates toolbox (toolbox/both modes)
4. **podman_service** -- Creates systemd Quadlet for vLLM (service/both modes)
5. **llamacpp_service** -- Downloads GGUF model, creates systemd Quadlet for llama.cpp (llamacpp mode)
6. **model_cache** -- Downloads HuggingFace model weights (skipped in llamacpp mode)
7. **openwebui_ui** -- Deploys Open WebUI container (if enabled, auto-connects to active backend)
8. **verify** -- Runs post-deployment health checks

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

## Step 8 -- Start Using Your Model

### llama.cpp Mode

The llama.cpp server starts automatically via systemd after deployment:

```bash
# Check status
systemctl --user status llamacpp-server

# View logs
mise run logs:llamacpp

# Switch model profiles
mise run deploy:llamacpp          # 122B (default)
mise run deploy:llamacpp:coder    # Coder 30B
mise run deploy:llamacpp:fast     # Fast 35B
mise run deploy:llamacpp:nemotron # Nemotron Nano 30B
mise run deploy:llamacpp:super    # Nemotron Super 120B
mise run deploy:llamacpp:minimax  # MiniMax-M2.7 229B

# Test the API
curl http://localhost:8080/v1/models
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```

### vLLM Toolbox Mode

```bash
toolbox enter vllm
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
vllm serve btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit \
  --enforce-eager --api-key local-dev-key
```

### vLLM Service Mode

```bash
systemctl --user status vllm-server
mise run logs:vllm
```

### Open WebUI

Browse to `http://<host>:3000`. The UI auto-connects to whichever backend is deployed (vLLM on port 8000 or llama.cpp on port 8080). If models don't appear, restart the container:

```bash
podman restart open-webui
```

---

## Next Steps

- [Performance Tuning](PERFORMANCE.md) -- Maximize tok/s on Strix Halo
- [Variables Reference](VARIABLES.md) -- All configuration options
- [Troubleshooting](TROUBLESHOOTING.md) -- Fix common issues
- [Architecture](ARCHITECTURE.md) -- Collection design and role structure
