# Architecture

## Collection Structure

```
ansible_collections/nerdsrun/strix_halo_vllm/
├── galaxy.yml
├── meta/runtime.yml
├── roles/
│   ├── host_prereqs/      # Hardware detection, packages, groups, image pull
│   ├── kernel_tuning/     # Optional kernel boot parameters (grubby)
│   ├── toolbox_mode/      # Interactive toolbox container
│   ├── podman_service/    # systemd Quadlet vLLM service
│   ├── model_cache/       # HuggingFace model prefetching
│   ├── openwebui_ui/      # Optional Open WebUI chat frontend
│   └── verify/            # Non-interactive deployment verification
├── playbooks/
│   ├── site.yml           # Full deployment
│   ├── toolbox.yml        # Toolbox-only
│   ├── service.yml        # Service-only
│   ├── verify.yml         # Verification only
│   ├── uninstall.yml      # Complete teardown
│   └── ui.yml             # Open WebUI management
├── docs/
│   ├── ARCHITECTURE.md    # This file
│   ├── VARIABLES.md       # Variable reference
│   └── TROUBLESHOOTING.md # Diagnostic guide
└── molecule/default/      # Test harness
```

## Role Dependency Graph

```
host_prereqs ──► kernel_tuning
     │
     ├──► toolbox_mode     (when mode=toolbox|both)
     │
     ├──► podman_service   (when mode=service|both)
     │
     ├──► model_cache      (always, when prefetch enabled)
     │
     ├──► openwebui_ui     (when ui_enabled)
     │
     └──► verify           (on-demand via tags)
```

## Mode Selection Logic

The `strix_halo_mode` variable controls which deployment path is taken:

- **toolbox**: `host_prereqs` -> `kernel_tuning` -> `toolbox_mode` -> `model_cache`
- **service**: `host_prereqs` -> `kernel_tuning` -> `podman_service` -> `model_cache`
- **both**: All of the above

## Fail-Fast Hardware Detection

`host_prereqs` reads `/sys/class/kfd/kfd/topology/nodes/*/properties` and extracts
`gfx_target_version`. If no node reports version `1151`, the play fails immediately
with an actionable error message showing what was detected vs. what is required.

## Idempotency Strategy

| Role | Strategy |
|---|---|
| host_prereqs | Package state checks, group membership append, conditional image pull |
| kernel_tuning | Compare `/proc/cmdline` against desired args, skip if present |
| toolbox_mode | `podman container inspect` to check existence, skip if present; firewalld rule when `firewall_open_vllm_port` |
| podman_service | Quadlet template with `notify` handlers, only restarts on change |
| model_cache | Check HuggingFace cache for existing models before downloading |
| openwebui_ui | `podman container exists` check, skip if present |
| verify | Read-only checks, no state changes |

## Security Posture

- All containers run rootless via user-level Podman
- systemd services use `scope: user` (no root)
- `seccomp=unconfined` is required for ROCm but other security layers remain active
- API key authentication on vLLM endpoints (configurable)
- Firewall rules opt-in only (`firewall_open_vllm_port: false` by default)

## Upgrade Strategy

- Set `strix_halo_toolbox_update: true` to recreate the toolbox with a new image
- Quadlet template changes trigger automatic service restart via handlers
- Re-running `site.yml` is safe -- no changes produced unless state has drifted
