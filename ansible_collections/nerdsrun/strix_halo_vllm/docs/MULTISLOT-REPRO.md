# Multi-slot collapse on hybrid models — reproduction record

Measured 2026-09-10 on the Strix Halo host. Kept here as a durable record and as
a ready-to-file upstream report; **not yet filed**. `PERFORMANCE.md` carries the
conclusions, this file carries the full repro so it does not have to be
rediscovered.

---

## Proposed issue title
Hybrid/recurrent models collapse and truncate at `--parallel > 1` (gfx1151/ROCm, `qwen35`); conventional models on the same build are fine

### Summary

On a hybrid linear-attention model (`qwen35` — Qwen3.8-27B, 49 of 65 layers Gated DeltaNet), running `llama-server` with `--parallel 2` under two concurrent requests causes decode to collapse to roughly 40% of aggregate single-slot throughput **and** generations to terminate after 1–13 tokens with `finish_reason: stop`.

The same binary, GPU and driver running a **conventional** MoE transformer of near-identical size at `--parallel 2` behaves correctly, and two **separate** `llama-server` processes on that same GPU sustain the full aggregate rate. So this is neither bandwidth nor the hardware — it is the in-process multi-sequence path on a recurrent architecture.

### Environment

- llama.cpp **b10707**, ROCm backend (TheRock 7.14.0), packaged by Lemonade Server 11.9.0
- AMD Ryzen AI Max+ 395, Radeon 8060S, **gfx1151**, 128 GB unified memory, Fedora 43
- Model: `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL (arch `qwen35`, `block_count` 65, 16 attention layers, `head_count_kv` 4, `key_length`=`value_length`=256, `ssm.inner_size` 6144)
- Control model: `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` UD-Q4_K_XL (conventional MoE, 16.8 GB vs 16.7 GB)

### Measurements

Offered load held at 2 for every row. Decode read from `llama-server`'s own `eval time` lines, not from the client. `max_tokens 200`, on a task that consumes all 200 when healthy.

| Configuration | Decode | Aggregate | Completions |
|---|---:|---:|---|
| `qwen35`, 1 process, `-np 1` | 21.9 tok/s | 21.9 | 6/6 full |
| **`qwen35`, 1 process, `-np 2`** | **4.4 tok/s** | **8.8** | **5 of 6 truncated (2-10 tokens)** |
| `qwen35`, **2 processes**, `-np 1` each | 11.3 / 11.0 | **22.3** | full, no truncation |
| Qwen3-Coder-30B, 1 process, `-np 1` | 16.5 tok/s | 16.5 | full |
| **Qwen3-Coder-30B, 1 process, `-np 2`** | 8.25 / 12.27 | **20.5** | **full, no truncation** |

Two independent processes conserve total throughput (22.3 ≈ 21.9), which is the expected behaviour for a bandwidth-bound part. `--parallel 2` on the same model delivers 8.8. The conventional model *gains* from `--parallel 2` (16.5 → 20.5) as intended.

Prefill is unaffected throughout (200–500 tok/s). Only decode collapses.

Server timings during a collapsed run:

```
task 27 | eval time =  1782.91 ms /  2 tokens ( 0.56 tok/s)
task 24 | eval time = 11406.34 ms /  6 tokens ( 0.44 tok/s)
task 37 | eval time =  6050.99 ms /  2 tokens ( 0.17 tok/s)
slot release: id 0 | task 27 | stop processing: n_tokens = 8908, truncated = 0
```

No error is logged; `truncated = 0` on every release and the server reports `backend_health: ready`.

### Ruled out

- **Speculative decoding** — `--spec-type none` measures 4.5 tok/s, indistinguishable from 4.4 with MTP.
- **The client** — reproduced with plain `curl`; `finish_reason: stop` comes from the server.
- **Prompt content / caching** — reproduced with unique prompts, with shared prefixes, with revisited prefixes, and with the prompt cache both cold and warm.
- **Bandwidth / hardware** — two separate processes reach full aggregate on the same GPU.
- **The slot count itself** — `--parallel 2` driven at offered load 1 is completely healthy (48/48 tokens, 21.2 tok/s).

### Reproduce

```bash
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf --ctx-size 524288 --parallel 2 \
  --jinja --metrics --port 8001
# then two simultaneous chat completions with ~9K-token prompts and max_tokens 200,
# asking for a detailed answer (a task that cannot be satisfied in a few tokens)
```

Compare against the same command with `--ctx-size 262144 --parallel 1`, and against the conventional model at `--parallel 2`.

### Note for anyone reproducing on ROCm

A bare `llama-server` launch silently loads the **CPU** backend (~25 tok/s prefill instead of ~360) unless the ROCm runtime is on `LD_LIBRARY_PATH`; `mem_info_gtt_used` never moves in that case and there is no warning. Confirm GPU residency with `rocm-smi --showpids` or per-pid `drm-total-gtt` in `/proc/<pid>/fdinfo/*` — the global GTT counter has no per-process attribution.
