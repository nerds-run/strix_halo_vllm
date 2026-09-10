#!/usr/bin/env python3
"""Unit tests for the Lemonade slot/cache sweep driver.

Everything tested here is pure computation — matrix generation, argv
verification, log parsing, memory arithmetic. The parts that talk to the live
server are deliberately not exercised: they need a box with a 17 GB model on it.

Run:  python3 -m unittest discover -s scripts/tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sweep_slots as ss  # noqa: E402


class TestBuildMatrix(unittest.TestCase):
    def test_phase_a_sweeps_cache_ram_at_one_slot(self):
        """Phase A isolates --cache-ram: slot count must stay at 1 throughout."""
        matrix = ss.build_matrix(cache_ram_values=[8192, 24576, 49152], slot_values=[1, 2, 3])
        phase_a = [c for c in matrix if c.phase == "A"]
        self.assertEqual([c.cache_ram_mib for c in phase_a], [8192, 24576, 49152])
        self.assertTrue(all(c.parallel == 1 for c in phase_a))

    def test_phase_b_sweeps_slots_at_fixed_cache_ram(self):
        """Phase B isolates slot count; cache-ram is pinned to the Phase A winner."""
        matrix = ss.build_matrix(
            cache_ram_values=[8192], slot_values=[1, 2, 3], best_cache_ram_mib=24576
        )
        phase_b = [c for c in matrix if c.phase == "B"]
        self.assertEqual([c.parallel for c in phase_b], [1, 2, 3])
        self.assertTrue(all(c.cache_ram_mib == 24576 for c in phase_b))

    def test_one_variable_changes_per_step(self):
        """The repo's tuning rule: exactly one variable moves between runs."""
        matrix = ss.build_matrix(cache_ram_values=[8192, 24576], slot_values=[1, 2])
        for prev, cur in zip(matrix, matrix[1:]):
            if prev.phase != cur.phase:
                continue
            changed = sum(
                [prev.cache_ram_mib != cur.cache_ram_mib, prev.parallel != cur.parallel]
            )
            self.assertEqual(changed, 1, f"{prev} -> {cur} changed {changed} variables")

    def test_phase_b_default_pool_keeps_every_slot_config_in_budget(self):
        """Pinning Phase B to the largest candidate pool would push 2 and 3
        slots over budget, so the sweep would silently test only 1 slot — the
        one thing it exists to compare. The default must be the largest pool
        that still leaves room for every slot count requested."""
        matrix = ss.build_matrix(
            cache_ram_values=[8192, 24576, 49152], slot_values=[1, 2, 3]
        )
        for c in [c for c in matrix if c.phase == "B"]:
            total = ss.estimate_gtt_gib(c.ctx_size, c.parallel, c.ctx_checkpoints) \
                + c.cache_ram_mib / 1024
            self.assertLessEqual(
                total, ss.BUDGET_GIB,
                f"{c.label} needs {total:.1f} GiB, over the {ss.BUDGET_GIB} GiB budget",
            )

    def test_explicit_best_cache_ram_still_wins(self):
        """An operator pinning the Phase A winner overrides the safe default."""
        matrix = ss.build_matrix(
            cache_ram_values=[8192], slot_values=[1], best_cache_ram_mib=24576
        )
        self.assertTrue(all(c.cache_ram_mib == 24576 for c in matrix if c.phase == "B"))

    def test_context_scales_with_slots_to_hold_full_window(self):
        matrix = ss.build_matrix(cache_ram_values=[8192], slot_values=[1, 2, 3])
        by_slots = {c.parallel: c.ctx_size for c in matrix if c.phase == "B"}
        self.assertEqual(by_slots[1], 262144)
        self.assertEqual(by_slots[2], 524288)
        self.assertEqual(by_slots[3], 786432)


class TestVerifyArgv(unittest.TestCase):
    """The silent-override guard, mirrored from the Ansible readback assert."""

    GOOD = (
        "llama-server -m model.gguf --ctx-size 524288 --port 8001 --jinja "
        "--spec-type draft-mtp --parallel 2 --ctx-checkpoints 8 --cache-ram 24576"
    )

    def test_accepts_matching_argv(self):
        ok, why = ss.verify_argv(self.GOOD, parallel=2, ctx_size=524288)
        self.assertTrue(ok, why)

    def test_rejects_duplicate_parallel_flag(self):
        """Catalog args merged the wrong way leave --parallel twice; llama.cpp
        takes the LAST one, so the server serves a count nobody asked for."""
        argv = self.GOOD + " --parallel 1"
        ok, why = ss.verify_argv(argv, parallel=2, ctx_size=524288)
        self.assertFalse(ok)
        self.assertIn("once", why.lower())

    def test_rejects_wrong_slot_count(self):
        ok, _ = ss.verify_argv(self.GOOD, parallel=3, ctx_size=786432)
        self.assertFalse(ok)

    def test_rejects_clamped_context(self):
        """Lemonade reports max_context_window 262144; if it clamps our 524288
        the run is invalid and must not be recorded as a 2-slot result."""
        argv = self.GOOD.replace("--ctx-size 524288", "--ctx-size 262144")
        ok, why = ss.verify_argv(argv, parallel=2, ctx_size=524288)
        self.assertFalse(ok)
        self.assertIn("ctx-size", why.lower())


class TestEvictionParsing(unittest.TestCase):
    LOG = """
2026-09-08 19:00:11 W srv alloc: - making room for prompt cache entry, removing oldest entry (size = 1471.291 MiB)
2026-09-08 19:00:44 W srv alloc: - making room for prompt cache entry, removing oldest entry (size = 1465.096 MiB)
2026-09-08 19:02:22 W srv alloc: - making room for prompt cache entry, removing oldest entry (size = 2559.390 MiB)
2026-09-08 19:03:26 I srv something else entirely
"""

    def test_counts_evictions(self):
        stats = ss.parse_cache_evictions(self.LOG)
        self.assertEqual(stats["count"], 3)

    def test_reports_entry_size_range(self):
        """Entry size drives how many fit in the pool — the whole point of the
        cache-ram axis."""
        stats = ss.parse_cache_evictions(self.LOG)
        self.assertAlmostEqual(stats["min_mib"], 1465.096, places=2)
        self.assertAlmostEqual(stats["max_mib"], 2559.390, places=2)

    def test_empty_log_is_zero_not_error(self):
        stats = ss.parse_cache_evictions("")
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["min_mib"])


class TestMemoryModel(unittest.TestCase):
    """Must agree with roles/lemonade_service/tasks/slot_args.yml — if these
    two drift, the guard and the benchmark are describing different machines."""

    def test_reproduces_measured_baseline(self):
        # Live host: 1 slot, ctx 262144, --ctx-checkpoints 32 -> 36.75 GiB GTT
        est = ss.estimate_gtt_gib(ctx_size=262144, parallel=1, ctx_checkpoints=32)
        self.assertGreaterEqual(est, 36.0)
        self.assertLessEqual(est, 40.0)

    def test_kv_scales_with_context_not_slots(self):
        a = ss.estimate_gtt_gib(ctx_size=524288, parallel=1, ctx_checkpoints=0)
        b = ss.estimate_gtt_gib(ctx_size=524288, parallel=2, ctx_checkpoints=0)
        self.assertAlmostEqual(a, b, places=6)

    def test_checkpoints_scale_with_slots_not_context(self):
        a = ss.estimate_gtt_gib(ctx_size=262144, parallel=1, ctx_checkpoints=32)
        b = ss.estimate_gtt_gib(ctx_size=262144, parallel=2, ctx_checkpoints=32)
        self.assertGreater(b - a, 4.0)

    def test_quad_full_window_exceeds_budget(self):
        est = ss.estimate_gtt_gib(ctx_size=1048576, parallel=4, ctx_checkpoints=8)
        self.assertGreater(est + 8, 85)


if __name__ == "__main__":
    unittest.main()


class TestConstantsMatchAnsible(unittest.TestCase):
    """The Python sweep and the Ansible deploy guard each carry their own copy
    of the model geometry. Both are anchored to the same hardware measurement,
    but nothing stops one file's constant from being updated without the other
    — at which point the guard and the benchmark describe different machines.
    This test makes the agreement enforced rather than merely asserted.
    """

    DEFAULTS = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "ansible_collections", "nerdsrun", "strix_halo_vllm",
        "roles", "lemonade_service", "defaults", "main.yml",
    )

    @classmethod
    def setUpClass(cls):
        import re as _re
        with open(cls.DEFAULTS) as fh:
            text = fh.read()
        cls.vals = {}
        for key in (
            "lemonade_slot_model_gib",
            "lemonade_slot_kv_kib_per_token",
            "lemonade_slot_ckpt_mib",
            "lemonade_gtt_budget_gib",
        ):
            m = _re.search(rf"^{key}:\s*([\d.]+)\s*$", text, _re.M)
            assert m, f"{key} not found in {cls.DEFAULTS}"
            cls.vals[key] = float(m.group(1))

    def test_model_size_matches(self):
        self.assertEqual(self.vals["lemonade_slot_model_gib"], ss.MODEL_GIB)

    def test_kv_per_token_matches(self):
        self.assertEqual(self.vals["lemonade_slot_kv_kib_per_token"], ss.KV_KIB_PER_TOKEN)

    def test_checkpoint_size_matches(self):
        self.assertEqual(self.vals["lemonade_slot_ckpt_mib"], ss.CKPT_MIB)

    def test_budget_matches(self):
        self.assertEqual(self.vals["lemonade_gtt_budget_gib"], ss.BUDGET_GIB)


class TestPrefixReuseWorkload(unittest.TestCase):
    """The cache-ram axis is only measurable with a workload that REUSES a long
    prefix — that is what the production traffic does (in=9463 over and over)
    and what an undersized pool destroys. A workload of unrelated prompts would
    show no difference between an 8 GiB and a 48 GiB pool.
    """

    def test_all_requests_share_the_prefix(self):
        w = ss.build_prefix_reuse_workload(prefix_tokens=2000, n_requests=5)
        prefixes = {p[: len(w[0]) - 200] for p in w}
        self.assertEqual(len(prefixes), 1, "requests must share a common prefix")

    def test_suffixes_differ(self):
        w = ss.build_prefix_reuse_workload(prefix_tokens=2000, n_requests=5)
        self.assertEqual(len(set(w)), 5, "each request needs a distinct suffix")

    def test_request_count(self):
        self.assertEqual(len(ss.build_prefix_reuse_workload(1000, 8)), 8)

    def test_prefix_length_is_roughly_requested(self):
        """Rough is fine — we need the prefix big enough to dominate prefill,
        not an exact token count."""
        w = ss.build_prefix_reuse_workload(prefix_tokens=4000, n_requests=2)
        approx_tokens = len(w[0]) / 4
        self.assertGreater(approx_tokens, 3000)
        self.assertLess(approx_tokens, 6000)

    def test_cache_benefit_summary_flags_a_thrashing_pool(self):
        """First request pays full prefill; the rest should not. If they do,
        the pool is too small for the working set."""
        thrashing = ss.summarize_cache_benefit([26.4, 26.3, 26.5, 26.4])
        self.assertFalse(thrashing["cache_effective"])
        healthy = ss.summarize_cache_benefit([26.4, 0.8, 0.7, 0.9])
        self.assertTrue(healthy["cache_effective"])
        self.assertGreater(healthy["speedup"], 10)

    def test_single_request_is_inconclusive_not_a_crash(self):
        r = ss.summarize_cache_benefit([26.4])
        self.assertIsNone(r["cache_effective"])


class TestStreamDeltaExtraction(unittest.TestCase):
    """Qwen3.8 is a reasoning model: with a short max_tokens most of the stream
    arrives as `reasoning_content`, and only a token or two as `content`.
    Counting `content` alone made the baseline run report decode_tps None.
    """

    def test_counts_plain_content(self):
        chunk = {"choices": [{"delta": {"content": "hello"}}]}
        self.assertEqual(ss.extract_delta_text(chunk), "hello")

    def test_counts_reasoning_content(self):
        chunk = {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
        self.assertEqual(ss.extract_delta_text(chunk), "thinking...")

    def test_counts_both_when_present(self):
        chunk = {"choices": [{"delta": {"reasoning_content": "a", "content": "b"}}]}
        self.assertEqual(ss.extract_delta_text(chunk), "ab")

    def test_empty_delta_is_empty_string(self):
        self.assertEqual(ss.extract_delta_text({"choices": [{"delta": {}}]}), "")

    def test_malformed_chunk_does_not_raise(self):
        self.assertEqual(ss.extract_delta_text({}), "")


class TestWorkingSetExceedsPool(unittest.TestCase):
    """Reproducing the production thrash needs MORE distinct prefixes than the
    pool can hold. With one shared prefix an 8 GiB pool caches it fine and the
    cache-ram axis measures nothing — which is exactly what the first baseline
    run showed (5.6x speedup, no thrash).
    """

    def test_distinct_prefixes_are_distinct(self):
        w = ss.build_prefix_reuse_workload(1000, n_requests=8, distinct_prefixes=4)
        heads = {p[:2000] for p in w}
        self.assertEqual(len(heads), 4)

    def test_each_prefix_is_revisited(self):
        """A prefix must come back AFTER others have evicted it — that is the
        thrash. Visiting each once would only measure cold prefill."""
        w = ss.build_prefix_reuse_workload(1000, n_requests=8, distinct_prefixes=4)
        heads = [p[:2000] for p in w]
        self.assertEqual(heads[0], heads[4], "prefixes must cycle, not run in blocks")
        self.assertNotEqual(heads[0], heads[1])

    def test_defaults_to_single_prefix(self):
        w = ss.build_prefix_reuse_workload(1000, n_requests=4)
        self.assertEqual(len({p[:2000] for p in w}), 1)

    def test_estimates_pool_pressure(self):
        """4 prefixes at ~1.5 GiB each need ~6 GiB; the 8 GiB default holds it.
        8 prefixes do not — that is the config that reproduces production."""
        self.assertFalse(ss.pool_will_thrash(distinct_prefixes=4, cache_ram_mib=8192))
        self.assertTrue(ss.pool_will_thrash(distinct_prefixes=8, cache_ram_mib=8192))
        self.assertFalse(ss.pool_will_thrash(distinct_prefixes=8, cache_ram_mib=49152))


class TestPhaseFilter(unittest.TestCase):
    """Phases are approved separately, so they have to be runnable separately
    without the other phase's configs sneaking in."""

    def test_filter_a_keeps_only_phase_a(self):
        m = ss.build_matrix([8192, 24576], [1, 2])
        self.assertTrue(all(c.phase == "A" for c in ss.filter_phase(m, "A")))
        self.assertEqual(len(ss.filter_phase(m, "A")), 2)

    def test_filter_b_keeps_only_phase_b(self):
        m = ss.build_matrix([8192, 24576], [1, 2])
        self.assertTrue(all(c.phase == "B" for c in ss.filter_phase(m, "B")))

    def test_ab_keeps_everything(self):
        m = ss.build_matrix([8192, 24576], [1, 2])
        self.assertEqual(len(ss.filter_phase(m, "AB")), len(m))


class TestConcurrencyControl(unittest.TestCase):
    """Comparing 1 slot at concurrency 1 against 2 slots at concurrency 2
    conflates two variables: it measures 'second slot' and 'second concurrent
    request' together. To isolate the slot, both configs must be driven at the
    SAME offered load — the load level where a slot could possibly help.
    """

    def test_defaults_to_the_slot_count(self):
        cfg = ss.SweepConfig("B", 2, 524288, 24576)
        self.assertEqual(ss.resolve_concurrency(cfg, None), 2)

    def test_override_pins_load_across_configs(self):
        one = ss.SweepConfig("B", 1, 262144, 24576)
        two = ss.SweepConfig("B", 2, 524288, 24576)
        self.assertEqual(ss.resolve_concurrency(one, 2), 2)
        self.assertEqual(ss.resolve_concurrency(two, 2), 2)

    def test_override_of_one_is_honoured_not_treated_as_falsy(self):
        cfg = ss.SweepConfig("B", 3, 786432, 8192)
        self.assertEqual(ss.resolve_concurrency(cfg, 1), 1)
