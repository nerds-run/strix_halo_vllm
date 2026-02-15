# Contributing

Thanks for your interest in contributing to **nerdsrun/amdllmv**!

## Getting Started

1. Fork and clone the repository
2. Run `mise run bootstrap` to install the toolchain
3. Create a feature branch from `main`

## Development Workflow

```bash
# Lint
mise run lint

# Run Molecule tests
mise run test

# Deploy to your target
mise run deploy:toolbox
```

## Submitting Changes

1. Keep commits focused -- one logical change per commit
2. Run `mise run lint` before pushing
3. Open a pull request against `main`
4. Describe **what** changed and **why**

## Code Style

- Follow existing Ansible conventions (FQCN modules, `become` over `sudo`)
- Use `ansible_facts['...']` instead of `ansible_*` top-level variables
- Tag tasks with the role name for selective execution
- Keep roles idempotent -- re-running should produce no changes

## Reporting Issues

Open an issue with:
- What you expected vs. what happened
- Output of `mise run verify` (if applicable)
- Host OS and hardware details
