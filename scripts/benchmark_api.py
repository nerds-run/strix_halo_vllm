#!/usr/bin/env python3
"""Automated performance benchmark for vLLM OpenAI-compatible API.

Measures tok/s, TTFT, and total latency across prompt sizes and concurrency
levels.  Outputs JSON + Markdown reports to benchmark_results/.

Dependencies: aiohttp (install via `mise run bootstrap`)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:
    print(
        "ERROR: aiohttp is not installed.\n"
        "  Run:  mise run bootstrap\n"
        "  Or:   pip install 'aiohttp>=3.9'"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Benchmark prompts
# ---------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "short": (
        "Explain the key differences between TCP and UDP protocols. "
        "Cover reliability, ordering, speed, and typical use cases for each."
    ),
    "medium": (
        "The following is a technical overview of modern CPU cache hierarchies:\n\n"
        "Modern processors use a multi-level cache hierarchy to bridge the gap "
        "between fast CPU cores and relatively slow main memory. The L1 cache, "
        "split into separate instruction (L1i) and data (L1d) caches, sits "
        "closest to each core and typically offers 32-64 KB per core with "
        "access latencies of 1-4 cycles. The L2 cache, usually unified and "
        "private per core, ranges from 256 KB to 1 MB with 10-20 cycle "
        "latency. The L3 cache (Last Level Cache) is shared across all cores, "
        "ranges from 4 MB to 128 MB in modern server processors, and has "
        "30-70 cycle latency.\n\n"
        "Cache coherence protocols like MESI and MOESI ensure that all cores "
        "see a consistent view of memory despite having private caches. When "
        "one core modifies a cache line, the protocol invalidates or updates "
        "copies in other cores' caches. This introduces coherence traffic on "
        "the interconnect, which can become a bottleneck in highly parallel "
        "workloads. Directory-based protocols scale better than snooping for "
        "large core counts.\n\n"
        "Cache replacement policies determine which lines to evict when the "
        "cache is full. While LRU (Least Recently Used) is theoretically "
        "optimal for temporal locality, practical implementations use pseudo-LRU "
        "approximations due to the high cost of tracking exact access order. "
        "Some modern processors use adaptive policies that detect access "
        "patterns and switch between LRU and frequency-based eviction.\n\n"
        "Prefetching is another critical optimization. Hardware prefetchers "
        "detect stride patterns in memory access and speculatively load data "
        "before it is needed. Software prefetch instructions allow programmers "
        "to hint at future accesses, though their effectiveness varies by "
        "workload. Overly aggressive prefetching can pollute caches and "
        "waste bandwidth.\n\n"
        "Non-uniform memory access (NUMA) architectures add another dimension "
        "to the memory hierarchy. In multi-socket systems, each processor has "
        "its own local memory controller. Accessing local memory is faster "
        "than accessing remote memory attached to another socket. Operating "
        "systems and applications must be NUMA-aware to achieve optimal "
        "performance, using techniques like memory interleaving, local "
        "allocation, and thread-to-core pinning.\n\n"
        "Task: Summarize the above passage in 3-4 concise bullet points, "
        "highlighting the most important concepts and trade-offs."
    ),
    "long": (
        "You are an expert code reviewer. Review the following Python module "
        "for correctness, performance, and style issues. Provide specific "
        "line-by-line feedback.\n\n"
        "```python\n"
        "import os\n"
        "import sys\n"
        "import json\n"
        "import time\n"
        "import hashlib\n"
        "import logging\n"
        "import threading\n"
        "from pathlib import Path\n"
        "from dataclasses import dataclass, field\n"
        "from typing import Optional, Dict, List, Any, Tuple\n"
        "from concurrent.futures import ThreadPoolExecutor, as_completed\n\n"
        "logger = logging.getLogger(__name__)\n\n\n"
        "@dataclass\n"
        "class CacheEntry:\n"
        "    key: str\n"
        "    value: Any\n"
        "    created_at: float = field(default_factory=time.time)\n"
        "    access_count: int = 0\n"
        "    ttl: float = 3600.0\n"
        "    size_bytes: int = 0\n\n"
        "    @property\n"
        "    def is_expired(self) -> bool:\n"
        "        return (time.time() - self.created_at) > self.ttl\n\n"
        "    def touch(self) -> None:\n"
        "        self.access_count += 1\n\n\n"
        "class LRUCache:\n"
        "    def __init__(self, max_size: int = 1000, max_memory_mb: float = 512):\n"
        "        self._store: Dict[str, CacheEntry] = {}\n"
        "        self._lock = threading.RLock()\n"
        "        self._max_size = max_size\n"
        "        self._max_memory = max_memory_mb * 1024 * 1024\n"
        "        self._current_memory = 0\n"
        "        self._hits = 0\n"
        "        self._misses = 0\n"
        "        self._evictions = 0\n\n"
        "    def get(self, key: str) -> Optional[Any]:\n"
        "        with self._lock:\n"
        "            entry = self._store.get(key)\n"
        "            if entry is None:\n"
        "                self._misses += 1\n"
        "                return None\n"
        "            if entry.is_expired:\n"
        "                self._remove(key)\n"
        "                self._misses += 1\n"
        "                return None\n"
        "            entry.touch()\n"
        "            self._hits += 1\n"
        "            return entry.value\n\n"
        "    def put(self, key: str, value: Any, ttl: float = 3600.0) -> None:\n"
        "        size = sys.getsizeof(value)\n"
        "        with self._lock:\n"
        "            if key in self._store:\n"
        "                self._remove(key)\n"
        "            while (len(self._store) >= self._max_size or\n"
        "                   self._current_memory + size > self._max_memory):\n"
        "                if not self._store:\n"
        "                    break\n"
        "                self._evict_one()\n"
        "            entry = CacheEntry(\n"
        "                key=key, value=value, ttl=ttl, size_bytes=size\n"
        "            )\n"
        "            self._store[key] = entry\n"
        "            self._current_memory += size\n\n"
        "    def _remove(self, key: str) -> None:\n"
        "        entry = self._store.pop(key, None)\n"
        "        if entry:\n"
        "            self._current_memory -= entry.size_bytes\n\n"
        "    def _evict_one(self) -> None:\n"
        "        if not self._store:\n"
        "            return\n"
        "        oldest_key = min(\n"
        "            self._store,\n"
        "            key=lambda k: (\n"
        "                self._store[k].access_count,\n"
        "                self._store[k].created_at\n"
        "            )\n"
        "        )\n"
        "        self._remove(oldest_key)\n"
        "        self._evictions += 1\n\n"
        "    def clear(self) -> None:\n"
        "        with self._lock:\n"
        "            self._store.clear()\n"
        "            self._current_memory = 0\n\n"
        "    @property\n"
        "    def stats(self) -> Dict[str, Any]:\n"
        "        total = self._hits + self._misses\n"
        "        hit_rate = self._hits / total if total > 0 else 0.0\n"
        "        return {\n"
        "            'size': len(self._store),\n"
        "            'memory_mb': self._current_memory / (1024 * 1024),\n"
        "            'hits': self._hits,\n"
        "            'misses': self._misses,\n"
        "            'hit_rate': round(hit_rate, 4),\n"
        "            'evictions': self._evictions,\n"
        "        }\n\n\n"
        "class FileBackedCache(LRUCache):\n"
        "    def __init__(self, cache_dir: str, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self._cache_dir = Path(cache_dir)\n"
        "        self._cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "        self._load_from_disk()\n\n"
        "    def _key_to_path(self, key: str) -> Path:\n"
        "        hashed = hashlib.sha256(key.encode()).hexdigest()\n"
        "        return self._cache_dir / f'{hashed}.json'\n\n"
        "    def _load_from_disk(self) -> None:\n"
        "        loaded = 0\n"
        "        for path in self._cache_dir.glob('*.json'):\n"
        "            try:\n"
        "                data = json.loads(path.read_text())\n"
        "                entry = CacheEntry(**data)\n"
        "                if not entry.is_expired:\n"
        "                    self._store[entry.key] = entry\n"
        "                    self._current_memory += entry.size_bytes\n"
        "                    loaded += 1\n"
        "                else:\n"
        "                    path.unlink()\n"
        "            except (json.JSONDecodeError, TypeError, KeyError):\n"
        "                logger.warning('Corrupt cache file: %s', path)\n"
        "                path.unlink()\n"
        "        logger.info('Loaded %d entries from disk cache', loaded)\n\n"
        "    def put(self, key: str, value: Any, ttl: float = 3600.0) -> None:\n"
        "        super().put(key, value, ttl)\n"
        "        entry = self._store.get(key)\n"
        "        if entry:\n"
        "            path = self._key_to_path(key)\n"
        "            path.write_text(json.dumps({\n"
        "                'key': entry.key,\n"
        "                'value': entry.value,\n"
        "                'created_at': entry.created_at,\n"
        "                'access_count': entry.access_count,\n"
        "                'ttl': entry.ttl,\n"
        "                'size_bytes': entry.size_bytes,\n"
        "            }))\n\n"
        "    def _remove(self, key: str) -> None:\n"
        "        path = self._key_to_path(key)\n"
        "        if path.exists():\n"
        "            path.unlink()\n"
        "        super()._remove(key)\n\n\n"
        "def process_batch(\n"
        "    items: List[Dict[str, Any]],\n"
        "    cache: LRUCache,\n"
        "    workers: int = 4\n"
        ") -> List[Tuple[str, Any]]:\n"
        "    results = []\n\n"
        "    def process_single(item):\n"
        "        key = item.get('id', '')\n"
        "        cached = cache.get(key)\n"
        "        if cached is not None:\n"
        "            return (key, cached)\n"
        "        value = item.get('data')\n"
        "        transformed = {\n"
        "            'processed': True,\n"
        "            'timestamp': time.time(),\n"
        "            'data': value,\n"
        "            'checksum': hashlib.md5(\n"
        "                json.dumps(value, sort_keys=True).encode()\n"
        "            ).hexdigest()\n"
        "        }\n"
        "        cache.put(key, transformed)\n"
        "        return (key, transformed)\n\n"
        "    with ThreadPoolExecutor(max_workers=workers) as pool:\n"
        "        futures = {pool.submit(process_single, item): item for item in items}\n"
        "        for future in as_completed(futures):\n"
        "            try:\n"
        "                results.append(future.result())\n"
        "            except Exception as exc:\n"
        "                item = futures[future]\n"
        "                logger.error('Failed to process %s: %s', item.get('id'), exc)\n\n"
        "    return results\n"
        "```\n\n"
        "Provide a structured review covering:\n"
        "1. Thread safety issues\n"
        "2. Performance concerns (especially in eviction and batch processing)\n"
        "3. API design and type annotation completeness\n"
        "4. Error handling gaps\n"
        "5. Suggestions for improvement"
    ),
}

PROMPT_LABELS: dict[str, str] = {
    "short": "Short (~50 tok)",
    "medium": "Medium (~500 tok)",
    "long": "Long (~2000 tok)",
}

CONCURRENCY_LEVELS = [1, 2, 4, 8]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    error: str | None = None


@dataclass
class AggregatedStats:
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0


@dataclass
class ConcurrencyResult:
    tokens_per_sec: AggregatedStats = field(default_factory=AggregatedStats)
    ttft_ms: AggregatedStats = field(default_factory=AggregatedStats)
    total_latency_ms: AggregatedStats = field(default_factory=AggregatedStats)
    errors: int = 0


# ---------------------------------------------------------------------------
# Hardware metadata
# ---------------------------------------------------------------------------


def collect_hardware_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": "unknown",
        "cpu_count": os.cpu_count() or 0,
        "memory_total_gb": 0,
    }

    # CPU model
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    meta["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    # Total memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    meta["memory_total_gb"] = round(kb / (1024 * 1024), 1)
                    break
    except OSError:
        pass

    return meta


# ---------------------------------------------------------------------------
# SSE streaming request
# ---------------------------------------------------------------------------


async def streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send a streaming chat completion request and measure performance."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    result = RequestResult()
    t_start = time.perf_counter()
    first_token_time: float | None = None
    server_tokens: int | None = None

    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:200]}"
                result.total_latency_ms = (time.perf_counter() - t_start) * 1000
                return result

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Check for content delta (TTFT)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content and first_token_time is None:
                        first_token_time = time.perf_counter()

                # Check for usage in final chunk
                usage = chunk.get("usage")
                if usage and "completion_tokens" in usage:
                    server_tokens = usage["completion_tokens"]

    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.error = str(exc)
        result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        return result

    t_end = time.perf_counter()
    result.total_latency_ms = (t_end - t_start) * 1000

    if first_token_time is not None:
        result.ttft_ms = (first_token_time - t_start) * 1000
        decode_time = t_end - first_token_time
    else:
        # No content tokens received
        result.ttft_ms = result.total_latency_ms
        decode_time = 0.0

    if server_tokens is not None:
        result.tokens_generated = server_tokens
    else:
        result.tokens_generated = 0

    if result.tokens_generated > 0 and decode_time > 0:
        result.tokens_per_sec = result.tokens_generated / decode_time
    else:
        result.tokens_per_sec = 0.0

    return result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def run_concurrent_batch(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
) -> list[RequestResult]:
    """Run `concurrency` simultaneous streaming requests."""
    tasks = [
        streaming_request(session, base_url, api_key, model, prompt, max_tokens)
        for _ in range(concurrency)
    ]
    return await asyncio.gather(*tasks)


def aggregate(values: list[float]) -> AggregatedStats:
    if not values:
        return AggregatedStats()
    return AggregatedStats(
        mean=round(statistics.mean(values), 2),
        min=round(min(values), 2),
        max=round(max(values), 2),
    )


async def detect_model(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
) -> str:
    """Auto-detect the model served by vLLM via GET /v1/models."""
    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                print(f"ERROR: GET {url} returned HTTP {resp.status}")
                print("Is vLLM running? Try: mise run logs:vllm")
                sys.exit(1)
            data = await resp.json()
            models = data.get("data", [])
            if not models:
                print("ERROR: vLLM returned no models.")
                sys.exit(1)
            model_id = models[0]["id"]
            return model_id
    except aiohttp.ClientError as exc:
        print(f"ERROR: Cannot connect to vLLM at {base_url}")
        print(f"  {exc}")
        print("Is vLLM running? Try: mise run logs:vllm")
        sys.exit(1)


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full benchmark suite and return results dict."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Detect model
        print("==> Detecting model...")
        model = await detect_model(session, args.base_url, args.api_key)
        print(f"    Model: {model}")

        # Warmup
        print("==> Warmup request (absorbs TunableOp first-run penalty)...")
        warmup = await streaming_request(
            session, args.base_url, args.api_key, model,
            "Say hello in one sentence.", args.max_tokens,
        )
        if warmup.error:
            print(f"    WARNING: Warmup failed: {warmup.error}")
        else:
            print(f"    Warmup: {warmup.tokens_generated} tokens, "
                  f"{warmup.tokens_per_sec:.1f} tok/s")

        # Collect hardware metadata
        hardware = collect_hardware_metadata()

        # Run benchmarks
        results: dict[str, Any] = {}
        total_combos = len(PROMPTS) * len(CONCURRENCY_LEVELS)
        combo_num = 0

        for prompt_name, prompt_text in PROMPTS.items():
            prompt_results: dict[str, Any] = {"concurrency": {}}

            for concurrency in CONCURRENCY_LEVELS:
                combo_num += 1
                label = PROMPT_LABELS[prompt_name]
                print(f"==> [{combo_num}/{total_combos}] {label}, "
                      f"concurrency={concurrency}, "
                      f"iterations={args.iterations}")

                all_tps: list[float] = []
                all_ttft: list[float] = []
                all_latency: list[float] = []
                total_errors = 0

                for i in range(args.iterations):
                    batch = await run_concurrent_batch(
                        session, args.base_url, args.api_key, model,
                        prompt_text, args.max_tokens, concurrency,
                    )
                    for r in batch:
                        if r.error:
                            total_errors += 1
                        else:
                            all_tps.append(r.tokens_per_sec)
                            all_ttft.append(r.ttft_ms)
                            all_latency.append(r.total_latency_ms)

                cr = ConcurrencyResult(
                    tokens_per_sec=aggregate(all_tps),
                    ttft_ms=aggregate(all_ttft),
                    total_latency_ms=aggregate(all_latency),
                    errors=total_errors,
                )

                if all_tps:
                    print(f"    tok/s: {cr.tokens_per_sec.mean:.1f} "
                          f"(min={cr.tokens_per_sec.min:.1f}, "
                          f"max={cr.tokens_per_sec.max:.1f}), "
                          f"TTFT: {cr.ttft_ms.mean:.0f}ms, "
                          f"errors: {cr.errors}")
                else:
                    print(f"    All requests failed ({cr.errors} errors)")

                prompt_results["concurrency"][str(concurrency)] = {
                    "tokens_per_sec": {
                        "mean": cr.tokens_per_sec.mean,
                        "min": cr.tokens_per_sec.min,
                        "max": cr.tokens_per_sec.max,
                    },
                    "ttft_ms": {
                        "mean": cr.ttft_ms.mean,
                        "min": cr.ttft_ms.min,
                        "max": cr.ttft_ms.max,
                    },
                    "total_latency_ms": {
                        "mean": cr.total_latency_ms.mean,
                        "min": cr.total_latency_ms.min,
                        "max": cr.total_latency_ms.max,
                    },
                    "errors": cr.errors,
                }

            results[prompt_name] = prompt_results

    return {
        "schema_version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": model,
            "max_tokens": args.max_tokens,
            "iterations": args.iterations,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "hardware": hardware,
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_markdown(data: dict[str, Any]) -> str:
    """Generate a Markdown report from benchmark results."""
    meta = data["metadata"]
    results = data["results"]
    lines: list[str] = []

    lines.append(f"# vLLM Benchmark Report")
    lines.append("")
    lines.append(f"- **Timestamp:** {meta['timestamp']}")
    lines.append(f"- **Model:** {meta['model_id']}")
    lines.append(f"- **Max tokens:** {meta['max_tokens']}")
    lines.append(f"- **Iterations:** {meta['iterations']}")
    hw = meta["hardware"]
    lines.append(f"- **Host:** {hw['hostname']}")
    lines.append(f"- **CPU:** {hw['cpu_model']}")
    lines.append(f"- **Memory:** {hw['memory_total_gb']} GB")
    lines.append("")

    # Summary table (concurrency=1 only)
    lines.append("## Summary (Concurrency = 1)")
    lines.append("")
    lines.append("| Prompt Size | tok/s | TTFT (ms) | Latency (ms) | Errors |")
    lines.append("|---|---|---|---|---|")
    for prompt_name in PROMPTS:
        c1 = results.get(prompt_name, {}).get("concurrency", {}).get("1", {})
        tps = c1.get("tokens_per_sec", {}).get("mean", 0)
        ttft = c1.get("ttft_ms", {}).get("mean", 0)
        lat = c1.get("total_latency_ms", {}).get("mean", 0)
        errs = c1.get("errors", 0)
        label = PROMPT_LABELS[prompt_name]
        lines.append(f"| {label} | {tps:.1f} | {ttft:.0f} | {lat:.0f} | {errs} |")
    lines.append("")

    # Detailed tables per prompt size
    for prompt_name in PROMPTS:
        label = PROMPT_LABELS[prompt_name]
        lines.append(f"## {label} - Detailed Results")
        lines.append("")
        lines.append("| Concurrency | tok/s (mean) | tok/s (min) | tok/s (max) "
                      "| TTFT ms (mean) | Latency ms (mean) | Errors |")
        lines.append("|---|---|---|---|---|---|---|")
        conc_data = results.get(prompt_name, {}).get("concurrency", {})
        for c in CONCURRENCY_LEVELS:
            cd = conc_data.get(str(c), {})
            tps = cd.get("tokens_per_sec", {})
            ttft = cd.get("ttft_ms", {})
            lat = cd.get("total_latency_ms", {})
            errs = cd.get("errors", 0)
            lines.append(
                f"| {c} "
                f"| {tps.get('mean', 0):.1f} "
                f"| {tps.get('min', 0):.1f} "
                f"| {tps.get('max', 0):.1f} "
                f"| {ttft.get('mean', 0):.0f} "
                f"| {lat.get('mean', 0):.0f} "
                f"| {errs} |"
            )
        lines.append("")

    # Concurrency scaling table
    lines.append("## Concurrency Scaling (Total Throughput)")
    lines.append("")
    lines.append("Aggregate tok/s across all concurrent requests at each level.")
    lines.append("")
    header = "| Prompt Size |"
    sep = "|---|"
    for c in CONCURRENCY_LEVELS:
        header += f" C={c} tok/s |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)
    for prompt_name in PROMPTS:
        label = PROMPT_LABELS[prompt_name]
        row = f"| {label} |"
        conc_data = results.get(prompt_name, {}).get("concurrency", {})
        for c in CONCURRENCY_LEVELS:
            cd = conc_data.get(str(c), {})
            mean_tps = cd.get("tokens_per_sec", {}).get("mean", 0)
            # Total throughput = per-request tok/s * concurrency
            total = mean_tps * c
            row += f" {total:.1f} |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM OpenAI-compatible API performance",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default="local-dev-key",
        help="API key (default: local-dev-key)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per completion (default: 128)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Iterations per (prompt, concurrency) combo (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for reports (default: benchmark_results/)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  vLLM Performance Benchmark")
    print("=" * 60)
    print(f"  Base URL:    {args.base_url}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Output dir:  {args.output_dir}")
    print("=" * 60)
    print()

    data = await run_benchmark(args)

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = data["metadata"]["model_id"].replace("/", "_")
    base_name = f"benchmark_{model_slug}_{ts}"

    json_path = out_dir / f"{base_name}.json"
    md_path = out_dir / f"{base_name}.md"

    json_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\n==> JSON report: {json_path}")

    md_content = generate_markdown(data)
    md_path.write_text(md_content)
    print(f"==> Markdown report: {md_path}")

    # Print summary to stdout
    print()
    print(md_content)


if __name__ == "__main__":
    asyncio.run(main())
