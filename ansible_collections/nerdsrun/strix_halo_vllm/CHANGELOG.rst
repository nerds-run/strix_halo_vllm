================================
nerdsrun.strix_halo_vllm Release
================================

v1.0.0
======

Release Summary
---------------
Initial release of the AMD Strix Halo vLLM Ansible collection.

Major Changes
-------------
- Added ``host_prereqs`` role for hardware detection and prerequisite installation.
- Added ``kernel_tuning`` role for AMD GPU kernel parameter optimization.
- Added ``toolbox_mode`` role for ROCm toolbox container deployment.
- Added ``podman_service`` role for vLLM systemd service via Podman Quadlet.
- Added ``model_cache`` role for HuggingFace model prefetching.
- Added ``openwebui_ui`` role for Open WebUI chat frontend.
- Added ``verify`` role for post-deployment validation.
- Playbooks: site, toolbox, service, verify, uninstall, ui.
