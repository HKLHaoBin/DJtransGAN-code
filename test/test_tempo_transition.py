import math
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


from djtransgan.process.tempo import (
    MAX_TEMPO_RATE_DELTA,
    apply_tempo_plan,
    build_tempo_plan,
    build_transition_window,
    estimate_local_tempo,
    resolve_tempo_match,
    source_to_target_frame,
    target_to_source_frame,
)
from djtransgan.process.beat import estimate_bpm
from djtransgan.process.process import (
    POSTPROCESS_CROSSFADE_SECONDS,
    postprocess,
    preprocess,
)
from djtransgan.process.sync import sync_cue


class LocalTempoTests(unittest.TestCase):
    def test_global_fallback_uses_robust_whole_track_intervals(self):
        intervals = [2.0 / 3.0] * 20 + [0.5] * 20 + [2.0 / 3.0] * 20
        beats = np.concatenate(([0.0], np.cumsum(intervals)))

        bpm = estimate_bpm(beats)

        self.assertTrue(math.isclose(bpm, 90.0, rel_tol=0.01))

    def test_uses_beats_near_cue_instead_of_track_middle(self):
        middle = [30.0 + 0.5 * i for i in range(16)]  # 120 BPM
        near_cue = [86.0 + (2.0 / 3.0) * i for i in range(13)]  # 90 BPM
        beats = middle + near_cue

        result = estimate_local_tempo(beats, cue_seconds=92.0, window_seconds=8.0)

        self.assertEqual(result.source, "cue_local")
        self.assertIsNone(result.reason)
        self.assertTrue(math.isclose(result.bpm, 90.0, rel_tol=0.01))

    def test_uses_explicit_fallback_when_cue_window_has_too_few_intervals(self):
        result = estimate_local_tempo(
            [1.0, 1.5, 2.0, 30.0],
            cue_seconds=30.0,
            window_seconds=2.0,
            fallback_bpm=123.0,
        )

        self.assertEqual(result.bpm, 123.0)
        self.assertEqual(result.source, "fallback")
        self.assertEqual(result.reason, "insufficient_local_intervals")
        self.assertEqual(result.intervals_used, 0)

    def test_filters_nonpositive_and_implausible_local_intervals(self):
        beats = [
            87.0,
            87.0 + 2.0 / 3.0,
            87.0 + 4.0 / 3.0,
            87.0 + 4.0 / 3.0,  # duplicate timestamp
            87.01 + 4.0 / 3.0,  # implausibly short interval
            89.01 + 4.0 / 3.0,  # implausibly long interval
            91.01,
            91.01 + 2.0 / 3.0,
            91.01 + 4.0 / 3.0,
            93.01,
        ]

        result = estimate_local_tempo(beats, cue_seconds=90.0, window_seconds=8.0)

        self.assertEqual(result.source, "cue_local")
        self.assertEqual(result.intervals_used, 6)
        self.assertTrue(math.isclose(result.bpm, 90.0, rel_tol=0.02))

    def test_rejects_nonfinite_or_nonpositive_fallback_bpm(self):
        for invalid in (float("nan"), 0.0, -90.0):
            with self.subTest(invalid=invalid):
                result = estimate_local_tempo(
                    [float("nan"), -1.0, 10.0],
                    cue_seconds=10.0,
                    window_seconds=2.0,
                    fallback_bpm=invalid,
                )

                self.assertIsNone(result.bpm)
                self.assertEqual(result.source, "unavailable")
                self.assertEqual(result.reason, "invalid_fallback_bpm")

    def test_filters_statistical_outlier_inside_plausible_bpm_bounds(self):
        beats = [20.0 + (2.0 / 3.0) * i for i in range(7)]
        beats.extend([25.2, 25.2 + 2.0 / 3.0, 25.2 + 4.0 / 3.0])

        result = estimate_local_tempo(beats, cue_seconds=23.0, window_seconds=8.0)

        self.assertEqual(result.intervals_used, 8)
        self.assertTrue(math.isclose(result.bpm, 90.0, rel_tol=0.01))


class TempoMatchTests(unittest.TestCase):
    def test_half_double_candidates_choose_rate_closest_to_native_speed(self):
        result = resolve_tempo_match(prev_bpm=70.0, next_bpm=140.0)

        self.assertEqual(result.raw_rate, 0.5)
        self.assertEqual(result.normalized_rate, 1.0)
        self.assertIn(
            (result.prev_multiplier, result.next_multiplier),
            {(2.0, 1.0), (1.0, 0.5)},
        )
        self.assertIsNone(result.reason)

    def test_invalid_local_bpm_disables_rate_resolution(self):
        for prev_bpm, next_bpm in ((float("nan"), 120.0), (120.0, 0.0), (-1.0, 90.0)):
            with self.subTest(prev_bpm=prev_bpm, next_bpm=next_bpm):
                result = resolve_tempo_match(prev_bpm=prev_bpm, next_bpm=next_bpm)

                self.assertIsNone(result.raw_rate)
                self.assertIsNone(result.normalized_rate)
                self.assertEqual(result.reason, "invalid_local_bpm")


class TempoPlanTests(unittest.TestCase):
    def test_clamps_start_rate_and_smoothly_returns_to_native_speed(self):
        plan = build_tempo_plan(prev_bpm=90.0, next_bpm=120.0, source_frames=48_000)

        self.assertEqual(MAX_TEMPO_RATE_DELTA, 0.08)
        self.assertEqual(plan.raw_rate, 0.75)
        self.assertEqual(plan.normalized_rate, 0.75)
        self.assertTrue(math.isclose(plan.applied_start_rate, 0.92))
        self.assertTrue(plan.clamped)
        rates = [rate for _, rate in plan.rate_keyframes]
        self.assertTrue(math.isclose(rates[0], 0.92))
        self.assertTrue(math.isclose(rates[-1], 1.0))
        self.assertTrue(all(left <= right for left, right in zip(rates, rates[1:])))
        self.assertEqual(plan.keyframes[0], (0, 0))
        self.assertEqual(plan.keyframes[-1][0], 48_000)
        self.assertTrue(
            all(
                x0 < x1 and y0 < y1
                for (x0, y0), (x1, y1) in zip(plan.keyframes, plan.keyframes[1:])
            )
        )
        # Integrating dy/dx=1/rate retains the early slow-down at the end;
        # the incorrect y=x/r(x) shortcut would end at exactly source_frames.
        self.assertGreater(plan.keyframes[-1][1], 48_000)

    def test_invalid_bpm_builds_disabled_identity_plan_with_reason(self):
        plan = build_tempo_plan(
            prev_bpm=float("nan"),
            next_bpm=120.0,
            source_frames=1_000,
        )

        self.assertFalse(plan.enabled)
        self.assertEqual(plan.reason, "invalid_local_bpm")
        self.assertEqual(plan.applied_start_rate, 1.0)
        self.assertEqual(plan.keyframes, ((0, 0), (1_000, 1_000)))

    def test_hold_is_shortened_when_needed_so_final_local_rate_is_native(self):
        plan = build_tempo_plan(
            prev_bpm=90.0,
            next_bpm=120.0,
            source_frames=10,
            hold_source_frames=100,
        )

        self.assertTrue(math.isclose(plan.rate_keyframes[0][1], 0.92))
        self.assertTrue(math.isclose(plan.rate_keyframes[-1][1], 1.0))

    def test_rejects_invalid_maximum_rate_delta(self):
        for invalid in (float("nan"), -0.01, 1.0):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "max_delta"):
                    build_tempo_plan(120.0, 100.0, 1_000, max_delta=invalid)

    def test_identity_plan_skips_rubberband_and_does_not_alias_input(self):
        audio = torch.arange(32, dtype=torch.float32).unsqueeze(0)
        original = audio.clone()
        plan = build_tempo_plan(prev_bpm=120.0, next_bpm=120.0, source_frames=32)

        with patch("djtransgan.process.tempo.pyrb.timemap_stretch") as stretch:
            output = apply_tempo_plan(audio, plan, sample_rate=44_100)

        stretch.assert_not_called()
        self.assertTrue(torch.equal(audio, original))
        self.assertTrue(torch.equal(output, original))
        self.assertIsNot(output, audio)

    def test_nonidentity_plan_calls_timemap_once_and_preserves_tensor_layout(self):
        audio = torch.stack(
            (
                torch.linspace(-1.0, 1.0, 32, dtype=torch.float32),
                torch.linspace(1.0, -1.0, 32, dtype=torch.float32),
            )
        )
        original = audio.clone()
        plan = build_tempo_plan(prev_bpm=90.0, next_bpm=120.0, source_frames=32)

        def fake_timemap(samples, sample_rate, keyframes, rbargs=None):
            self.assertEqual(samples.shape, (32, 2))
            self.assertEqual(sample_rate, 44_100)
            self.assertEqual(keyframes, list(plan.keyframes))
            self.assertEqual(rbargs, {"--fine": ""})
            return np.zeros((plan.keyframes[-1][1], 2), dtype=samples.dtype)

        with patch(
            "djtransgan.process.tempo.pyrb.timemap_stretch",
            side_effect=fake_timemap,
        ) as stretch:
            output = apply_tempo_plan(audio, plan, sample_rate=44_100)

        stretch.assert_called_once()
        self.assertEqual(output.shape, (2, plan.keyframes[-1][1]))
        self.assertEqual(output.dtype, audio.dtype)
        self.assertEqual(output.device, audio.device)
        self.assertTrue(torch.equal(audio, original))

    def test_source_target_coordinate_mapping_is_bounded_and_reversible(self):
        plan = build_tempo_plan(prev_bpm=100.0, next_bpm=120.0, source_frames=10_000)

        for source, target in plan.keyframes:
            self.assertEqual(source_to_target_frame(source, plan.keyframes), float(target))
            self.assertEqual(target_to_source_frame(target, plan.keyframes), float(source))
        self.assertEqual(source_to_target_frame(-1, plan.keyframes), 0.0)
        self.assertEqual(
            source_to_target_frame(20_000, plan.keyframes),
            float(plan.keyframes[-1][1]),
        )
        self.assertEqual(target_to_source_frame(-1, plan.keyframes), 0.0)
        self.assertEqual(
            target_to_source_frame(20_000, plan.keyframes),
            float(plan.keyframes[-1][0]),
        )

    def test_transition_window_stretches_only_local_next_material_once(self):
        audio = torch.arange(200, dtype=torch.float32).unsqueeze(0)
        original = audio.clone()

        def fake_timemap(samples, sample_rate, keyframes, rbargs=None):
            self.assertLess(len(samples), audio.size(-1))
            target_frames = keyframes[-1][1]
            positions = np.linspace(0, len(samples) - 1, target_frames).round().astype(int)
            return samples[positions]

        with patch(
            "djtransgan.process.tempo.pyrb.timemap_stretch",
            side_effect=fake_timemap,
        ) as stretch:
            transition = build_transition_window(
                audio,
                source_start_frame=10,
                target_start_frame=20,
                model_frames=100,
                prev_bpm=130.0,
                next_bpm=100.0,
                hold_source_frames=16,
                sample_rate=44_100,
            )

        stretch.assert_called_once()
        self.assertEqual(transition.audio.shape, (1, 100))
        self.assertEqual(transition.source_start_frame, 10)
        self.assertLess(transition.source_end_frame, audio.size(-1))
        self.assertEqual(transition.target_start_frame, 20)
        self.assertEqual(transition.target_end_frame, 100)
        self.assertTrue(math.isclose(transition.plan.rate_keyframes[-1][1], 1.0))
        self.assertTrue(torch.equal(audio, original))

    def test_empty_transition_at_track_end_is_bounded_and_skips_external_stretch(self):
        audio = torch.arange(5, dtype=torch.float32).unsqueeze(0)

        with patch("djtransgan.process.tempo.pyrb.timemap_stretch") as stretch:
            transition = build_transition_window(
                audio,
                source_start_frame=99,
                target_start_frame=100,
                model_frames=100,
                prev_bpm=90.0,
                next_bpm=120.0,
                sample_rate=44_100,
            )

        stretch.assert_not_called()
        self.assertEqual(transition.audio.shape, (1, 100))
        self.assertEqual(transition.source_start_frame, 5)
        self.assertEqual(transition.source_end_frame, 5)
        self.assertFalse(transition.plan.enabled)
        self.assertEqual(transition.plan.reason, "empty_transition")


class PreprocessTempoIntegrationTests(unittest.TestCase):
    def test_match_and_align_share_one_local_timemap_and_keep_original_next(self):
        prev_audio = torch.linspace(-0.8, 0.8, 200).unsqueeze(0)
        next_audio = torch.linspace(0.7, -0.7, 200).unsqueeze(0)
        original_next = next_audio.clone()
        prev_beats = np.arange(0.0, 20.0, 0.5)  # 120 BPM
        next_beats = np.arange(0.0, 20.0, 0.6)  # 100 BPM
        prev_downbeats = np.arange(0.0, 20.0, 2.0)
        next_downbeats = np.arange(0.0, 20.0, 2.4)

        def fake_timemap(samples, sample_rate, keyframes, rbargs=None):
            self.assertLess(len(samples), next_audio.size(-1))
            target_frames = keyframes[-1][1]
            positions = np.linspace(0, len(samples) - 1, target_frames).round().astype(int)
            return samples[positions]

        with (
            patch("djtransgan.process.process.settings.SR", 10),
            patch("djtransgan.process.process.settings.N_TIME", 10),
            patch("djtransgan.process.process.settings.CUE_BAR", 2),
            patch(
                "djtransgan.process.process.estimate_beat",
                side_effect=[
                    ((4, 4), 120.0, prev_beats, prev_downbeats),
                    ((4, 4), 100.0, next_beats, next_downbeats),
                ],
            ),
            patch("djtransgan.process.process.normalize", side_effect=lambda audio: audio.clone()),
            patch(
                "djtransgan.process.sync.time_stretch",
                side_effect=AssertionError("legacy constant-rate stretch must not run"),
            ),
            patch(
                "djtransgan.process.tempo.pyrb.timemap_stretch",
                side_effect=fake_timemap,
            ) as stretch,
        ):
            (pair_audio, timestamps), (pair_audio_for_g, _), meta = preprocess(
                prev_audio,
                next_audio,
                prev_cue=8.0,
                next_cue=9.6,
                match_bpm=True,
                align_cue=True,
                max_tempo_rate_delta=0.2,
            )

        stretch.assert_called_once()
        self.assertTrue(torch.equal(next_audio, original_next))
        self.assertTrue(torch.equal(pair_audio[1], original_next))
        self.assertEqual(pair_audio_for_g[1].shape[-1], 100)
        self.assertLess(timestamps[1][1], next_audio.size(-1))
        self.assertTrue(math.isclose(meta["prev_local_bpm"], 120.0))
        self.assertTrue(math.isclose(meta["next_local_bpm"], 100.0))
        self.assertTrue(meta["tempo_transition_applied"])
        self.assertTrue(math.isclose(meta["applied_start_rate"], 1.2))
        self.assertTrue(math.isclose(meta["max_delta"], 0.2))
        json.dumps(meta, allow_nan=False)

    def test_disabling_both_tempo_controls_keeps_native_baseline_and_skips_stretch(self):
        prev_audio = torch.linspace(-0.8, 0.8, 200).unsqueeze(0)
        next_audio = torch.linspace(0.7, -0.7, 200).unsqueeze(0)
        prev_beats = np.arange(0.0, 20.0, 0.5)
        next_beats = np.arange(0.0, 20.0, 0.6)
        prev_downbeats = np.arange(0.0, 20.0, 2.0)
        next_downbeats = np.arange(0.0, 20.0, 2.4)

        with (
            patch("djtransgan.process.process.settings.SR", 10),
            patch("djtransgan.process.process.settings.N_TIME", 10),
            patch("djtransgan.process.process.settings.CUE_BAR", 2),
            patch(
                "djtransgan.process.process.estimate_beat",
                side_effect=[
                    ((4, 4), 120.0, prev_beats, prev_downbeats),
                    ((4, 4), 100.0, next_beats, next_downbeats),
                ],
            ),
            patch("djtransgan.process.process.normalize", side_effect=lambda audio: audio.clone()),
            patch("djtransgan.process.tempo.pyrb.timemap_stretch") as stretch,
        ):
            (pair_audio, _), (pair_audio_for_g, _), meta = preprocess(
                prev_audio,
                next_audio,
                prev_cue=8.0,
                next_cue=9.6,
                match_bpm=False,
                align_cue=False,
            )

        stretch.assert_not_called()
        self.assertTrue(torch.equal(pair_audio[0], prev_audio))
        self.assertTrue(torch.equal(pair_audio[1], next_audio))
        self.assertEqual(pair_audio_for_g[0].shape[-1], 100)
        self.assertEqual(pair_audio_for_g[1].shape[-1], 100)
        self.assertFalse(meta["tempo_transition_applied"])
        self.assertEqual(meta["tempo_disabled_reason"], "tempo_controls_disabled")

    def test_postprocess_suffix_comes_from_original_next_source_anchor(self):
        prev_audio = -torch.arange(200, dtype=torch.float32).unsqueeze(0)
        next_audio = (1_000 + torch.arange(150, dtype=torch.float32)).unsqueeze(0)
        mix_audio = torch.zeros((1, 100), dtype=torch.float32)
        timestamps = [[20, 120], [10, 50]]
        cue = torch.tensor([0.2, 0.8], dtype=torch.float32)

        with (
            patch("djtransgan.process.process.settings.SR", 1_000),
            patch("djtransgan.process.process.settings.N_TIME", 0.1),
        ):
            output, _ = postprocess(
                mix_audio,
                [prev_audio, next_audio],
                timestamps,
                cue,
            )

        fade_frames = int(round(POSTPROCESS_CROSSFADE_SECONDS * 1_000))
        expected_length = 20 + 100 + (150 - 50) - fade_frames
        self.assertEqual(output.size(-1), expected_length)
        exact_suffix_start = 20 + 100
        self.assertTrue(
            torch.equal(
                output[:, exact_suffix_start:],
                next_audio[:, 50 + fade_frames:],
            )
        )


class LegacyCompatibilityTests(unittest.TestCase):
    def test_sync_cue_uses_source_over_target_rubberband_rate(self):
        prev_audio = torch.zeros((1, 30), dtype=torch.float32)
        next_audio = torch.zeros((1, 20), dtype=torch.float32)
        observed_rates = []

        def fake_stretch(audio, rate):
            observed_rates.append(rate)
            return torch.zeros((1, 20), dtype=audio.dtype)

        with (
            patch("djtransgan.process.sync.settings.SR", 1),
            patch("djtransgan.process.sync.time_stretch", side_effect=fake_stretch),
        ):
            sync_cue(prev_audio, next_audio, [0.0, 20.0], [0.0, 10.0])

        self.assertEqual(observed_rates, [0.5])


if __name__ == "__main__":
    unittest.main()
