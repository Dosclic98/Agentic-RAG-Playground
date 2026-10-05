import json
import sys
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from generation_metrics import GenerationMetrics


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


def chunk(content=None, thinking=None, tool_calls=None, done=False, count=None, duration=None):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, thinking=thinking, tool_calls=tool_calls),
        done=done, eval_count=count, eval_duration=duration,
    )


class GenerationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.metrics = GenerationMetrics(self.clock)
        self.metrics.begin_turn()

    def test_fresh_turn_is_known_zero_and_snapshot_is_frozen(self):
        snapshot = self.metrics.snapshot()
        self.assertEqual((snapshot.tokens, snapshot.estimated, snapshot.elapsed, snapshot.rate),
                         (0, False, 0.0, None))
        with self.assertRaises(FrozenInstanceError):
            snapshot.tokens = 1

    def test_initial_generation_without_explicit_turn_starts_wall_timer(self):
        metrics = GenerationMetrics(self.clock)
        self.clock.value = 10.0
        metrics.begin_generation()
        self.clock.value = 12.0
        metrics.progress(chunk(content="abcdefgh"))
        snapshot = metrics.snapshot()
        self.assertEqual((snapshot.tokens, snapshot.estimated, snapshot.elapsed, snapshot.rate),
                         (2, True, 2.0, None))

    def test_estimates_are_independent_of_text_chunk_boundaries(self):
        for pieces in (("abcd",), ("a", "b", "c", "d"), ("ab", "cd")):
            with self.subTest(pieces=pieces):
                self.metrics.begin_turn()
                self.metrics.begin_generation()
                for piece in pieces:
                    self.metrics.progress(chunk(content=piece))
                self.assertEqual(self.metrics.snapshot().tokens, 1)
                self.assertTrue(self.metrics.snapshot().estimated)

    def test_thinking_and_content_share_one_cumulative_estimate(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(thinking="abc"))
        self.metrics.progress(chunk(thinking="d", content="efgh"))
        self.assertEqual(self.metrics.snapshot().tokens, 2)

    def test_tool_only_generation_counts_serialized_calls_without_ollama_dependency(self):
        calls = [{"function": {"name": "search_pdf", "arguments": {"query": "revenue"}}},
                 {"function": {"name": "calculate", "arguments": {"expression": "2+2"}}}]
        characters = sum(len(json.dumps(call, ensure_ascii=False, separators=(",", ":")))
                         for call in calls)
        self.metrics.begin_generation()
        self.metrics.progress(chunk(tool_calls=calls))
        combined = self.metrics.snapshot()
        self.assertEqual(combined.tokens, (characters + 3) // 4)
        self.assertTrue(combined.estimated)
        self.metrics.begin_generation()
        for call in calls:
            self.metrics.progress(chunk(tool_calls=[call]))
        self.assertEqual(self.metrics.snapshot().tokens, combined.tokens)
        self.metrics.progress(chunk(done=True, count=17))
        self.metrics.end_generation()
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (17, False))

    def test_duck_typed_tool_models_and_namespace_objects_are_supported(self):
        class Tool:
            def model_dump(self):
                return {"function": {"name": "lookup", "arguments": {"query": "résumé"}}}

        for call in (Tool(), SimpleNamespace(function=SimpleNamespace(name="lookup", arguments={})),
                     {"function": {"name": "lookup", "arguments": {}}}):
            with self.subTest(call=type(call).__name__):
                self.metrics.begin_generation()
                self.metrics.progress(chunk(tool_calls=[call]))
                self.assertGreater(self.metrics.snapshot().tokens, 0)

    def test_final_count_can_reconcile_estimate_downward(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="abcdefgh" * 10))
        self.assertEqual(self.metrics.snapshot().tokens, 20)
        self.metrics.progress(chunk(done=True, count=3))
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (3, False))
        self.metrics.end_generation()
        self.metrics.end_generation()
        self.metrics.progress(chunk(done=True, count=3))
        self.assertEqual(self.metrics.snapshot().tokens, 3)

    def test_intermediate_cumulative_counts_are_not_added_to_final_counter(self):
        self.metrics.begin_generation()
        for count in (1, 2, 3):
            self.metrics.progress(chunk(content="word", count=count))
        self.metrics.progress(chunk(done=True, count=4))
        self.metrics.end_generation()
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (4, False))

    def test_multiple_rounds_accumulate_exact_counts_and_active_estimate(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=6, duration=1_000_000_000))
        self.metrics.end_generation()
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="abcdefgh"))
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (8, True))
        self.metrics.progress(chunk(done=True, count=3, duration=1_000_000_000))
        self.metrics.end_generation()
        self.clock.value = 10.0
        self.metrics.finish_turn(generated_tokens=9)
        snapshot = self.metrics.snapshot()
        self.assertEqual((snapshot.tokens, snapshot.estimated, snapshot.elapsed, snapshot.rate),
                         (9, False, 10.0, 4.5))

    def test_missing_count_and_measured_zero_remain_distinct(self):
        for count, estimated in ((None, True), (0, False), (-1, True), (True, True), (1.5, True)):
            with self.subTest(count=count):
                self.metrics.begin_turn()
                self.metrics.begin_generation()
                self.metrics.progress(chunk(done=True, count=count))
                self.metrics.end_generation()
                self.metrics.finish_turn()
                self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated),
                                 (0, estimated))

    def test_missing_completed_count_keeps_later_total_approximate(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="abcd", done=True))
        self.metrics.end_generation()
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=3))
        self.metrics.end_generation()
        self.metrics.finish_turn()
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (4, True))

    def test_tool_wait_advances_elapsed_and_has_no_active_generation_rate(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=6))
        self.metrics.end_generation()
        self.clock.value = 2.0
        before = self.metrics.snapshot()
        self.clock.value = 8.0
        after = self.metrics.snapshot()
        self.assertEqual((before.tokens, after.tokens), (6, 6))
        self.assertEqual((before.rate, after.rate), (None, None))
        self.assertEqual(after.elapsed, 8.0)
        self.assertFalse(after.estimated)

    def test_retry_discards_abandoned_attempt_but_preserves_completed_calls(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=6))
        self.metrics.end_generation()
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="x" * 40))
        self.metrics.end_generation()
        self.assertEqual(self.metrics.snapshot().tokens, 16)
        self.metrics.begin_generation()
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (6, True))
        self.metrics.progress(chunk(done=True, count=3))
        self.metrics.end_generation()
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (9, False))

    def test_interruption_retains_provisional_count_and_freezes_elapsed_and_rate(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(thinking="abcdefgh"))
        self.metrics.end_generation()
        self.clock.value = 2.0
        self.metrics.finish_turn()
        before = self.metrics.snapshot()
        self.clock.value = 100.0
        self.metrics.finish_turn(generated_tokens=999)
        after = self.metrics.snapshot()
        self.assertEqual(before, after)
        self.assertEqual((after.tokens, after.estimated, after.elapsed, after.rate), (2, True, 2.0, None))

    def test_final_absolute_count_override_replaces_estimates(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="x" * 40, done=True))
        self.metrics.end_generation()
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="abcd"))
        self.metrics.finish_turn(generated_tokens=7)
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (7, False))

    def test_failure_before_generation_is_unknown_zero_and_explicit_zero_is_exact(self):
        self.metrics.finish_turn()
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (0, True))
        self.metrics.begin_turn()
        self.metrics.finish_turn(generated_tokens=0)
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (0, False))

    def test_active_generation_is_unknown_even_before_first_text(self):
        self.metrics.begin_generation()
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().estimated), (0, True))
        self.metrics.finish_turn()
        self.assertTrue(self.metrics.snapshot().estimated)

    def test_reset_discards_counts_and_frozen_time(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=6))
        self.metrics.end_generation()
        self.clock.value = 10.0
        self.metrics.finish_turn()
        self.clock.value = 100.0
        self.metrics.begin_turn()
        snapshot = self.metrics.snapshot()
        self.assertEqual((snapshot.tokens, snapshot.estimated, snapshot.elapsed, snapshot.rate),
                         (0, False, 0.0, None))

    def test_explicit_start_and_nonpositive_elapsed_avoid_division_by_zero(self):
        self.metrics.begin_turn(started=10.0)
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=5))
        self.metrics.end_generation()
        self.clock.value = 9.0
        snapshot = self.metrics.snapshot()
        self.assertEqual((snapshot.elapsed, snapshot.rate), (0.0, None))
        self.clock.value = 12.0
        self.assertEqual((self.metrics.snapshot().elapsed, self.metrics.snapshot().rate), (2.0, None))

    def test_default_clock_observes_dynamic_monotonic_patch(self):
        metrics = GenerationMetrics()
        with patch("generation_metrics.time.monotonic", return_value=10.0):
            metrics.begin_turn()
        with patch("generation_metrics.time.monotonic", return_value=12.0):
            self.assertEqual(metrics.snapshot().elapsed, 2.0)

    def test_live_speed_starts_at_first_emission_and_excludes_long_startup(self):
        self.metrics.begin_generation()
        self.clock.value = 100.0
        self.metrics.progress(chunk())
        self.metrics.progress(chunk(content="x" * 100))
        self.assertIsNone(self.metrics.snapshot().rate)
        self.clock.value = 101.0
        self.metrics.progress(chunk(content="x" * 100))
        snapshot = self.metrics.snapshot()
        self.assertEqual((snapshot.tokens, snapshot.elapsed, snapshot.rate), (50, 101.0, 25.0))
        self.assertTrue(snapshot.rate_estimated)

    def test_rolling_window_ignores_older_bursts_and_reaches_zero_when_stale(self):
        self.metrics.begin_generation()
        self.clock.value = 10.0
        self.metrics.progress(chunk(content="x" * 100))
        self.clock.value = 10.5
        self.metrics.progress(chunk(content="x" * 200))
        self.assertEqual(self.metrics.snapshot().rate, 100.0)
        self.clock.value = 11.5
        self.metrics.progress(chunk(content="x" * 40))
        self.assertEqual(self.metrics.snapshot().rate, 10.0)
        self.clock.value = 12.5
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot.rate, 0.0)
        self.assertTrue(snapshot.rate_estimated)

    def test_same_clock_bursts_do_not_invent_a_rate_or_divide_by_zero(self):
        self.metrics.begin_generation()
        for _ in range(20):
            self.metrics.progress(chunk(content="abcd"))
        self.assertIsNone(self.metrics.snapshot().rate)
        self.clock.value = 0.5
        self.metrics.progress(chunk(content="abcd" * 5))
        self.assertEqual(self.metrics.snapshot().rate, 10.0)

    def test_boundary_anchor_excludes_deltas_before_the_window(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="abcd"))
        self.clock.value = 0.2
        self.metrics.progress(chunk(content="abcd" * 100))
        self.clock.value = 0.7
        self.metrics.progress(chunk(content="abcd" * 2))
        self.clock.value = 1.3
        self.metrics.progress(chunk(content="abcd" * 3))
        # The 100-token burst at 0.2 is outside (0.3, 1.3]. The retained
        # boundary baseline removes it before the denominator is clipped.
        self.assertEqual(self.metrics.snapshot().rate, 5.0)

    def test_live_rate_uses_fractional_character_estimates_not_rounded_counts(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="a"))
        self.clock.value = 0.5
        self.metrics.progress(chunk(content="b"))
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot.tokens, 1)
        self.assertEqual(snapshot.rate, 0.5)

    def test_known_count_can_have_an_independently_estimated_live_rate(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="abcd"))
        self.clock.value = 0.5
        self.metrics.progress(chunk(content="abcdefgh"))
        self.metrics.progress(chunk(done=True, count=100, duration=1_000_000_000))
        snapshot = self.metrics.snapshot()
        self.assertFalse(snapshot.estimated)
        self.assertTrue(snapshot.rate_estimated)
        self.assertEqual(snapshot.rate, 4.0)
        self.metrics.end_generation()
        self.assertIsNone(self.metrics.snapshot().rate)
        self.metrics.finish_turn()
        self.assertEqual((self.metrics.snapshot().rate, self.metrics.snapshot().rate_estimated),
                         (100.0, False))

    def test_thinking_and_tool_emissions_also_update_live_speed(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(thinking="abcd"))
        self.clock.value = 0.5
        call = {"function": {"name": "lookup", "arguments": {}}}
        characters = len(json.dumps(call, ensure_ascii=False, separators=(",", ":")))
        self.metrics.progress(chunk(tool_calls=[call]))
        self.assertEqual(self.metrics.snapshot().rate, characters / 4.0 / 0.5)

    def test_live_sampling_remains_bounded_with_many_distinct_emission_times(self):
        self.metrics.begin_generation()
        for index in range(2000):
            self.clock.value = index / 10000
            self.metrics.progress(chunk(content="abcd"))
        self.assertLessEqual(len(self.metrics._current.samples), 512)
        self.assertAlmostEqual(self.metrics.snapshot().rate, 10000.0)
        self.clock.value += 2.0
        self.assertEqual(self.metrics.snapshot().rate, 0.0)

    def test_final_average_is_weighted_by_generation_duration_and_ignores_wall_time(self):
        self.metrics.begin_generation()
        self.clock.value = 100.0
        self.metrics.progress(chunk(done=True, count=20, duration=1_000_000_000))
        self.metrics.end_generation()
        self.clock.value = 900.0
        self.assertIsNone(self.metrics.snapshot().rate)
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=90, duration=3_000_000_000))
        self.metrics.end_generation()
        self.clock.value = 1000.0
        self.metrics.finish_turn(generated_tokens=110)
        snapshot = self.metrics.snapshot()
        self.assertEqual((snapshot.tokens, snapshot.elapsed, snapshot.rate, snapshot.rate_estimated),
                         (110, 1000.0, 27.5, False))

    def test_final_speed_needs_valid_duration_on_every_completed_call(self):
        for duration in (None, 0, -1, True, 1.0, "1000000000"):
            with self.subTest(duration=duration):
                self.metrics.begin_turn()
                self.metrics.begin_generation()
                self.metrics.progress(chunk(done=True, count=20, duration=1_000_000_000))
                self.metrics.end_generation()
                self.metrics.begin_generation()
                self.metrics.progress(chunk(done=True, count=90, duration=duration))
                self.metrics.end_generation()
                self.metrics.finish_turn(generated_tokens=110)
                self.assertIsNone(self.metrics.snapshot().rate)
                self.assertFalse(self.metrics.snapshot().rate_estimated)

    def test_missing_final_count_cannot_be_repaired_by_absolute_override_for_speed(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="abcd", done=True, duration=1_000_000_000))
        self.metrics.end_generation()
        self.metrics.finish_turn(generated_tokens=50)
        snapshot = self.metrics.snapshot()
        self.assertEqual((snapshot.tokens, snapshot.estimated, snapshot.rate), (50, False, None))

    def test_conflicting_absolute_override_invalidates_paired_average(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=20, duration=1_000_000_000))
        self.metrics.end_generation()
        self.metrics.finish_turn(generated_tokens=30)
        self.assertEqual(self.metrics.snapshot().tokens, 30)
        self.assertIsNone(self.metrics.snapshot().rate)

    def test_real_zero_speed_requires_positive_measured_duration(self):
        for duration, rate in ((1_000_000_000, 0.0), (0, None), (None, None)):
            with self.subTest(duration=duration):
                self.metrics.begin_turn()
                self.metrics.begin_generation()
                self.metrics.progress(chunk(done=True, count=0, duration=duration))
                self.metrics.end_generation()
                self.metrics.finish_turn(generated_tokens=0)
                self.assertEqual(self.metrics.snapshot().rate, rate)
                self.assertFalse(self.metrics.snapshot().rate_estimated)

    def test_cumulative_intermediate_metadata_is_not_summed_for_final_speed(self):
        self.metrics.begin_generation()
        for count in (1, 2, 3):
            self.metrics.progress(chunk(content="abcd", count=count, duration=100_000_000))
        self.metrics.progress(chunk(done=True, count=4, duration=2_000_000_000))
        self.metrics.end_generation()
        self.metrics.end_generation()
        self.metrics.finish_turn()
        self.assertEqual((self.metrics.snapshot().tokens, self.metrics.snapshot().rate), (4, 2.0))

    def test_retry_restarts_live_samples_without_polluting_final_average(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=20, duration=1_000_000_000))
        self.metrics.end_generation()
        self.metrics.begin_generation()
        self.clock.value = 1.0
        self.metrics.progress(chunk(content="x" * 100))
        self.clock.value = 1.5
        self.metrics.progress(chunk(content="x" * 100))
        self.assertEqual(self.metrics.snapshot().rate, 50.0)
        self.metrics.end_generation()
        self.assertIsNone(self.metrics.snapshot().rate)
        self.metrics.begin_generation()
        self.clock.value = 100.0
        self.metrics.progress(chunk(content="x" * 100))
        self.assertIsNone(self.metrics.snapshot().rate)
        self.clock.value = 100.5
        self.metrics.progress(chunk(content="x" * 100))
        self.assertEqual(self.metrics.snapshot().rate, 50.0)
        self.metrics.progress(chunk(done=True, count=90, duration=3_000_000_000))
        self.metrics.end_generation()
        self.metrics.finish_turn(generated_tokens=110)
        self.assertEqual(self.metrics.snapshot().rate, 27.5)

    def test_interruption_invalidates_final_speed_even_with_previous_measured_calls(self):
        self.metrics.begin_generation()
        self.metrics.progress(chunk(done=True, count=20, duration=1_000_000_000))
        self.metrics.end_generation()
        self.metrics.begin_generation()
        self.metrics.progress(chunk(content="abcd"))
        self.clock.value = 0.5
        self.metrics.progress(chunk(content="abcdefgh"))
        self.assertEqual(self.metrics.snapshot().rate, 4.0)
        self.metrics.end_generation()
        self.metrics.finish_turn(generated_tokens=20)
        self.assertIsNone(self.metrics.snapshot().rate)
        self.clock.value = 100.0
        self.assertIsNone(self.metrics.snapshot().rate)

    def test_background_snapshots_remain_consistent_during_stream_updates(self):
        self.metrics.begin_generation()
        self.clock.value = 2.0
        entered, saw_progress, done = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def render():
            entered.set()
            while not done.is_set():
                snapshot = self.metrics.snapshot()
                if snapshot.rate is not None and snapshot.rate < 0:
                    errors.append(snapshot)
                if snapshot.tokens:
                    saw_progress.set()
                done.wait(0.001)

        reader = threading.Thread(target=render)
        reader.start()
        try:
            self.assertTrue(entered.wait(1))
            for _ in range(2000):
                self.metrics.progress(chunk(content="word"))
            self.assertTrue(saw_progress.wait(1))
            self.metrics.progress(chunk(done=True, count=1000, duration=2_000_000_000))
            self.metrics.end_generation()
            self.metrics.finish_turn()
        finally:
            done.set()
            reader.join(1)
        self.assertFalse(reader.is_alive())
        self.assertEqual(errors, [])
        snapshot = self.metrics.snapshot()
        self.assertEqual((snapshot.tokens, snapshot.estimated, snapshot.rate), (1000, False, 500.0))


if __name__ == "__main__":
    unittest.main()
