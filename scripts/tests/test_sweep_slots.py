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
