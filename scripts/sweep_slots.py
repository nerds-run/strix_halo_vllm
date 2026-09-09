#!/usr/bin/env python3
"""Sweep Lemonade slot count and prompt-cache size on the Strix Halo box.

Two phases, one variable at a time (the repo's tuning rule):

  Phase A  --cache-ram at 1 slot.  The live server logs
             "srv alloc: - making room for prompt cache entry, removing oldest"
           on essentially every request: entries are 1.3-2.6 GiB and llama.cpp's
           default pool is 8 GiB, so it holds three to five of them. This phase
           asks what the pool is worth before slot count is touched at all.

  Phase B  slot count at the Phase A winner.  Slots SUBDIVIDE context, so each
           step raises --ctx-size in lockstep to hold a full 262144-token
           window per request.

Every run is verified against the argv llama-server actually launched with,
never against the flag submitted: Lemonade merges our args with the catalog's,
the catalog already emits `--parallel 1`, and llama.cpp takes the LAST
occurrence of a repeated flag.

  ./scripts/sweep_slots.py --dry-run          # print the matrix, touch nothing
  ./scripts/sweep_slots.py --host 192.168.68.60
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

# --- Model geometry -------------------------------------------------------
# Qwen3.8-27B UD-Q4_K_XL, parsed from the GGUF header on hardware:
# arch qwen35, block_count 65 with 16 attention layers, head_count_kv 4,
# key_length = value_length = 256 -> 16 * 4 * 512 * 2 B = 64 KiB per token.
# Weights 16746 MiB + mmproj-F16 885 MiB = 17.22 GiB.
#
# These MUST agree with roles/lemonade_service/tasks/slot_args.yml. If they
# drift, the deploy guard and this benchmark describe different machines.
MODEL_GIB = 17.22
KV_KIB_PER_TOKEN = 64
CKPT_MIB = 149.6
CTX_PER_SLOT = 262144
DEFAULT_MODEL = "Qwen3.8-27B-GGUF"
DEFAULT_PORT = 13305
BUDGET_GIB = 85.0


@dataclass
class SweepConfig:
    phase: str
    parallel: int
    ctx_size: int
    cache_ram_mib: int
    ctx_checkpoints: int = 8

    @property
    def label(self) -> str:
        return f"{self.phase}:np{self.parallel}:cache{self.cache_ram_mib}"

    @property
    def llamacpp_args(self) -> str:
        # Each flag exactly once. See the module docstring.
        return (
            f"--parallel {self.parallel} "
            f"--ctx-checkpoints {self.ctx_checkpoints} "
            f"--cache-ram {self.cache_ram_mib}"
        )


def estimate_gtt_gib(ctx_size: int, parallel: int, ctx_checkpoints: int) -> float:
    """Weights + KV + recurrent-state checkpoints, in GiB.

    KV scales with TOTAL context; checkpoints scale with SLOT COUNT. Keeping
    them separate is what makes an over-budget result diagnosable.

    No compute-buffer term: it is driven by ubatch, not by slots or context,
    and it is why this reads slightly high against the 36.75 GiB measured at
    1 slot / 262144 / ckpt 32.
    """
    kv = ctx_size * KV_KIB_PER_TOKEN / (1024 * 1024)
    ckpt = parallel * ctx_checkpoints * CKPT_MIB / 1024
    return round(MODEL_GIB + kv + ckpt, 2)


def build_matrix(
    cache_ram_values: list[int],
    slot_values: list[int],
    best_cache_ram_mib: int | None = None,
    ctx_checkpoints: int = 8,
) -> list[SweepConfig]:
    """Phase A sweeps the cache pool at 1 slot; Phase B sweeps slots at the
    Phase A winner. Exactly one variable moves between consecutive runs."""
    matrix = [
        SweepConfig("A", 1, CTX_PER_SLOT, c, ctx_checkpoints) for c in cache_ram_values
    ]
    pinned = (
        best_cache_ram_mib
        if best_cache_ram_mib is not None
        else _largest_affordable_pool(cache_ram_values, slot_values, ctx_checkpoints)
    )
    matrix += [
        SweepConfig("B", n, n * CTX_PER_SLOT, pinned, ctx_checkpoints) for n in slot_values
    ]
    return matrix


def _largest_affordable_pool(
    cache_ram_values: list[int], slot_values: list[int], ctx_checkpoints: int
) -> int:
    """Pick the biggest candidate pool that still fits every requested slot count.

    The cache pool and the slots compete for the same unified memory, so simply
    pinning Phase B to the largest Phase A pool would push 2 and 3 slots over
    budget — leaving the sweep testing only 1 slot, the one comparison it
    exists to make. Falls back to the smallest candidate if nothing fits, and
    lets the budget guard skip whatever it must.
    """
    worst = max(slot_values)
    headroom = BUDGET_GIB - estimate_gtt_gib(worst * CTX_PER_SLOT, worst, ctx_checkpoints)
    affordable = [c for c in cache_ram_values if c / 1024 <= headroom]
    return max(affordable) if affordable else min(cache_ram_values)


def verify_argv(argv: str, parallel: int, ctx_size: int) -> tuple[bool, str]:
    """Check the argv llama-server was actually launched with.

    Order matters: a duplicated flag is reported as such rather than as a value
    mismatch, because the two have different causes and different fixes.
    """
    occurrences = len(re.findall(r"--parallel\b", argv))
    if occurrences == 0:
        return False, "--parallel absent from the launched argv"
    if occurrences > 1:
        return False, (
            f"--parallel appears {occurrences} times; it must appear exactly once "
            "or llama.cpp silently takes the last occurrence"
        )
    if not re.search(rf"--parallel\s+{parallel}\b", argv):
        got = re.search(r"--parallel\s+(\d+)", argv)
        return False, f"--parallel is {got.group(1) if got else '?'}, expected {parallel}"
    if not re.search(rf"--ctx-size\s+{ctx_size}\b", argv):
        got = re.search(r"--ctx-size\s+(\d+)", argv)
        return False, (
            f"--ctx-size is {got.group(1) if got else '?'}, expected {ctx_size} "
            "(Lemonade may have clamped it to the model's max_context_window)"
        )
    return True, "argv matches the requested configuration"


def parse_cache_evictions(log_text: str) -> dict:
    """Count prompt-cache evictions and the size range of evicted entries.

    Entry size is the number that decides how many fit in the pool, which is
    the entire point of the cache-ram axis.
    """
    sizes = [
        float(m)
        for m in re.findall(
            r"making room for prompt cache entry.*?size\s*=\s*([\d.]+)\s*MiB", log_text
        )
    ]
    return {
        "count": len(sizes),
        "min_mib": min(sizes) if sizes else None,
        "max_mib": max(sizes) if sizes else None,
        "mean_mib": round(sum(sizes) / len(sizes), 2) if sizes else None,
    }


# --- Live driver ----------------------------------------------------------

def _api(host: str, port: int, path: str, body: dict | None = None, timeout: int = 900):
    url = f"http://{host}:{port}/api/v1{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if body is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _ssh(host: str, key: str, cmd: str) -> str:
    out = subprocess.run(
        ["ssh", "-i", key, "-o", "BatchMode=yes", f"abanna@{host}", cmd],
        capture_output=True, text=True, timeout=120, env={"SSH_AUTH_SOCK": ""},
    )
    return out.stdout


def gtt_used_mib(host: str, key: str) -> int | None:
    raw = _ssh(host, key, "cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1")
    try:
        return int(raw.strip()) // (1024 * 1024)
    except (ValueError, AttributeError):
        return None


def apply_config(host: str, port: int, model: str, cfg: SweepConfig) -> tuple[bool, str, str]:
    """Push options, reload, then read back the launched argv."""
    _api(host, port, f"/models/{model}/options", {
        "ctx_size": cfg.ctx_size,
        "llamacpp_args": cfg.llamacpp_args,
        "merge_args": True,
    })
    _api(host, port, "/load", {"model_name": model})
    time.sleep(5)
    health = _api(host, port, "/health")
    argv = ""
    for m in health.get("all_models_loaded", []):
        if m.get("model_name") == model:
            argv = " ".join(m.get("launch_command", []))
            break
    ok, why = verify_argv(argv, cfg.parallel, cfg.ctx_size)
    return ok, why, argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.68.60")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ssh-key", default=".ssh/framework_fedora")
    ap.add_argument("--cache-ram", default="8192,24576,49152",
                    help="Phase A pool sizes in MiB")
    ap.add_argument("--slots", default="1,2,3", help="Phase B slot counts")
    ap.add_argument("--best-cache-ram", type=int, default=None,
                    help="Pin Phase B to this pool size instead of the max")
    ap.add_argument("--ctx-checkpoints", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the matrix and the memory estimates; touch nothing")
    ap.add_argument("--out", default="benchmark_results")
    args = ap.parse_args()

    matrix = build_matrix(
        [int(x) for x in args.cache_ram.split(",")],
        [int(x) for x in args.slots.split(",")],
        args.best_cache_ram,
        args.ctx_checkpoints,
    )

    print(f"{'phase':<6}{'slots':>6}{'ctx':>10}{'window':>9}{'cache':>8}{'est GTT':>10}{'+pool':>9}")
    print("-" * 60)
    over = []
    for c in matrix:
        est = estimate_gtt_gib(c.ctx_size, c.parallel, c.ctx_checkpoints)
        total = round(est + c.cache_ram_mib / 1024, 2)
        flag = ""
        if total > BUDGET_GIB:
            flag = "  OVER BUDGET"
            over.append(c)
        print(f"{c.phase:<6}{c.parallel:>6}{c.ctx_size:>10}{CTX_PER_SLOT:>9}"
              f"{c.cache_ram_mib:>8}{est:>10.2f}{total:>9.2f}{flag}")
    if over:
        print(f"\n{len(over)} configuration(s) exceed the {BUDGET_GIB} GiB budget and will be skipped.")
    matrix = [c for c in matrix if estimate_gtt_gib(c.ctx_size, c.parallel, c.ctx_checkpoints)
              + c.cache_ram_mib / 1024 <= BUDGET_GIB]

    if args.dry_run:
        print("\n--dry-run: nothing applied.")
        return 0

    print(f"\nEach step reloads {args.model}, interrupting in-flight requests.")
    results = []
    for i, cfg in enumerate(matrix, 1):
        print(f"\n[{i}/{len(matrix)}] {cfg.label}")
        before = gtt_used_mib(args.host, args.ssh_key)
        try:
            ok, why, argv = apply_config(args.host, args.port, args.model, cfg)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  APPLY FAILED: {e}")
            results.append({**asdict(cfg), "applied": False, "error": str(e)})
            continue
        print(f"  {'OK' if ok else 'MISMATCH'}: {why}")
        if not ok:
            results.append({**asdict(cfg), "applied": False, "error": why, "argv": argv})
            continue
        after = gtt_used_mib(args.host, args.ssh_key)
        results.append({
            **asdict(cfg), "applied": True, "argv": argv,
            "gtt_before_mib": before, "gtt_after_mib": after,
            "gtt_est_gib": estimate_gtt_gib(cfg.ctx_size, cfg.parallel, cfg.ctx_checkpoints),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"slot_sweep_{stamp}.json"
    path.write_text(json.dumps({"generated": stamp, "model": args.model, "runs": results}, indent=2))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
