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


# --- Prefix-reuse workload ------------------------------------------------

# One paragraph of filler, repeated to reach the requested prefix length. The
# content is irrelevant — what matters is that every request shares a long,
# identical prefix, because that is the shape of the traffic this box actually
# serves (the same ~9463-token prompt, over and over) and the shape an
# undersized prompt-cache pool destroys.
_FILLER = (
    "The deployment runs llama.cpp behind Lemonade on an AMD Ryzen AI Max 395 "
    "with 128 GB of unified memory, serving a hybrid linear-attention model "
    "where only sixteen of sixty-five layers carry a KV cache. "
)


# A cached entry for a ~9.5K-token prefix measures ~1.5 GiB on this model —
# taken from the eviction lines the live server logs, not from theory.
CACHE_ENTRY_GIB = 1.5


def build_prefix_reuse_workload(
    prefix_tokens: int, n_requests: int, distinct_prefixes: int = 1
) -> list[str]:
    """N prompts drawn from `distinct_prefixes` long prefixes, cycled.

    Cycled rather than blocked: a prefix has to come back AFTER the others have
    had a chance to evict it, which is what makes an undersized pool show up.
    Running each prefix's requests consecutively would only ever measure one
    cold prefill followed by guaranteed hits.

    Roughly 4 characters per token is close enough — the prefix only has to be
    long enough to dominate prefill, not to hit an exact count.
    """
    reps = max(1, (prefix_tokens * 4) // len(_FILLER))
    prefixes = [
        f"Document {d}. " + (_FILLER * reps) for d in range(distinct_prefixes)
    ]
    return [
        f"{prefixes[i % distinct_prefixes]}"
        f"\n\nQuestion {i}: summarise the document above in {i + 3} words."
        for i in range(n_requests)
    ]


def pool_will_thrash(distinct_prefixes: int, cache_ram_mib: int) -> bool:
    """Whether the working set is too big for the pool to hold.

    This is the production failure: entries are ~1.5 GiB, the default pool is
    8 GiB, so anything past ~5 live prefixes evicts on every request.
    """
    return distinct_prefixes * CACHE_ENTRY_GIB > cache_ram_mib / 1024


def extract_delta_text(chunk: dict) -> str:
    """Text from one streaming chunk, counting reasoning as well as content.

    Qwen3.8 emits most of a short response as `reasoning_content`; counting
    only `content` under-counts tokens badly and can report no decode at all.
    """
    try:
        delta = chunk["choices"][0]["delta"]
    except (KeyError, IndexError, TypeError):
        return ""
    return (delta.get("reasoning_content") or "") + (delta.get("content") or "")


def summarize_cache_benefit(ttfts: list[float]) -> dict:
    """Compare the first request's TTFT against the rest.

    The first pays a full prefill by definition. If the rest also do, the pool
    could not hold the entry between requests — which is the thrash already
    visible in production, not a property of the model.
    """
    if len(ttfts) < 2:
        return {"cache_effective": None, "first_s": ttfts[0] if ttfts else None,
                "rest_mean_s": None, "speedup": None}
    first = ttfts[0]
    rest = sum(ttfts[1:]) / len(ttfts[1:])
    speedup = (first / rest) if rest > 0 else float("inf")
    return {
        "cache_effective": speedup >= 2.0,
        "first_s": round(first, 2),
        "rest_mean_s": round(rest, 2),
        "speedup": round(speedup, 2),
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


def run_workload(
    host: str, port: int, model: str, prompts: list[str], max_tokens: int = 64,
    concurrency: int = 1,
) -> dict:
    """Send the prompts and measure TTFT plus decode rate for each.

    Streaming, because TTFT is the number the cache-ram axis moves and it is
    only observable on the first token, not on total latency.
    """
    import concurrent.futures as _cf

    def one(prompt: str) -> dict:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"http://{host}:{port}/api/v1/chat/completions",
            data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        ttft = None
        ntok = 0
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if extract_delta_text(chunk):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
        total = time.perf_counter() - t0
        decode = (ntok - 1) / (total - ttft) if ttft and total > ttft and ntok > 1 else 0.0
        return {"ttft_s": ttft or total, "total_s": total, "tokens": ntok,
                "decode_tps": round(decode, 2)}

    if concurrency > 1:
        with _cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            runs = list(ex.map(one, prompts))
    else:
        runs = [one(p) for p in prompts]

    ttfts = [r["ttft_s"] for r in runs]
    decodes = [r["decode_tps"] for r in runs if r["decode_tps"] > 0]
    return {
        "runs": runs,
        "ttft_first_s": round(ttfts[0], 2) if ttfts else None,
        "ttft_mean_s": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "decode_tps_mean": round(sum(decodes) / len(decodes), 2) if decodes else None,
        "cache": summarize_cache_benefit(ttfts),
    }


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
    ap.add_argument("--prefix-tokens", type=int, default=9500,
                    help="Shared-prefix length; defaults to the size the box actually serves")
    ap.add_argument("--requests", type=int, default=12,
                    help="Requests per config, cycled over the distinct prefixes")
    ap.add_argument("--distinct-prefixes", type=int, default=8,
                    help="Distinct long prefixes in the working set. The default "
                         "exceeds the 8 GiB pool (~1.5 GiB per entry), reproducing "
                         "the eviction thrash seen in production.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--baseline-only", action="store_true",
                    help="Benchmark the CURRENTLY LOADED config and exit. No reload, no eviction.")
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

    prompts = build_prefix_reuse_workload(
        args.prefix_tokens, args.requests, args.distinct_prefixes)
    if pool_will_thrash(args.distinct_prefixes, 8192):
        print(f"\nWorking set: {args.distinct_prefixes} prefixes x ~{CACHE_ENTRY_GIB} GiB "
              f"= ~{args.distinct_prefixes * CACHE_ENTRY_GIB:.1f} GiB — "
              f"exceeds the 8192 MiB default pool, as intended.")

    if args.baseline_only:
        # Measures whatever is loaded right now. No options written, no reload,
        # so nothing is evicted — this is the "before" number.
        health = _api(args.host, args.port, "/health")
        argv = ""
        for m in health.get("all_models_loaded", []):
            if m.get("model_name") == args.model:
                argv = " ".join(m.get("launch_command", []))
        print(f"Benchmarking the loaded config (no reload).\n  argv: {argv}\n")
        bench = run_workload(args.host, args.port, args.model, prompts,
                             args.max_tokens, concurrency=1)
        print(f"  TTFT first    : {bench['ttft_first_s']}s")
        print(f"  TTFT mean     : {bench['ttft_mean_s']}s")
        print(f"  decode tok/s  : {bench['decode_tps_mean']}")
        print(f"  cache verdict : {bench['cache']}")
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out / f"slot_baseline_{stamp}.json"
        path.write_text(json.dumps(
            {"generated": stamp, "model": args.model, "argv": argv,
             "gtt_used_mib": gtt_used_mib(args.host, args.ssh_key),
             "prefix_tokens": args.prefix_tokens, "requests": args.requests,
             "bench": bench}, indent=2))
        print(f"\nWrote {path}")
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
        bench = run_workload(args.host, args.port, args.model, prompts,
                             args.max_tokens, concurrency=cfg.parallel)
        print(f"  GTT {after} MiB (est {estimate_gtt_gib(cfg.ctx_size, cfg.parallel, cfg.ctx_checkpoints)} GiB)"
              f"  TTFT {bench['ttft_first_s']}s -> {bench['cache']['rest_mean_s']}s"
              f"  decode {bench['decode_tps_mean']} tok/s")
        results.append({
            **asdict(cfg), "applied": True, "argv": argv,
            "gtt_before_mib": before, "gtt_after_mib": after,
            "gtt_est_gib": estimate_gtt_gib(cfg.ctx_size, cfg.parallel, cfg.ctx_checkpoints),
            "bench": bench,
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
