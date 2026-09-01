import numpy as np
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

import torch
from djtransgan.config import settings
from djtransgan.utils import squeeze_dim
from djtransgan.process import filter_beat


def _patch_madmom_dbn_numpy():
    """madmom 0.16 + NumPy>=1.24: np.asarray(list_of_(path,score)) fails on jagged paths."""
    if getattr(DBNDownBeatTrackingProcessor.process, '_dj_patched', False):
        return
    _orig = DBNDownBeatTrackingProcessor.process

    def process(self, activations, **kwargs):
        real_asarray = np.asarray

        def asarray_safe(a, *args, **kw):
            try:
                return real_asarray(a, *args, **kw)
            except ValueError:
                if (
                    isinstance(a, (list, tuple))
                    and len(a)
                    and isinstance(a[0], (list, tuple))
                    and len(a[0]) == 2
                ):
                    # Stand-in so callers can do asarray(results)[:, 1]
                    return real_asarray([[0.0, float(x[1])] for x in a], dtype=float)
                raise

        np.asarray = asarray_safe
        try:
            return _orig(self, activations, **kwargs)
        finally:
            np.asarray = real_asarray

    process._dj_patched = True
    DBNDownBeatTrackingProcessor.process = process


_patch_madmom_dbn_numpy()


def estimate_beat(audio):
    try:

        if isinstance(audio, torch.Tensor):
            audio = squeeze_dim(audio).numpy()

        proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        act = RNNDownBeatProcessor()(audio)
        proc_res = proc(act)
        if proc_res is None or len(proc_res) == 0:
            raise RuntimeError('beat tracker returned empty result')
        beat = set(proc_res[:, 1])
        downbeat = proc_res[proc_res[:, 1] == 1, 0]

        sig = int(max(beat))
        beat_curve = proc_res[:, 0]
        if sig == 6:
            sig_d = 8
        else:
            sig_d = 4

        return (sig, sig_d), estimate_bpm(beat_curve), filter_beat(beat_curve, downbeat, sig_d), downbeat

    except Exception as e:
        print(e)
        return ()


def estimate_bpm(beat_curve):
    """Robust whole-track fallback retained for compatibility.

    Active matching uses cue-local tempo; this fallback no longer samples only
    the middle third of a track.
    """
    try:
        intervals = np.diff(np.asarray(beat_curve, dtype=np.float64).reshape(-1))
        intervals = intervals[
            np.isfinite(intervals)
            & (intervals >= 60.0 / 240.0)
            & (intervals <= 60.0 / 40.0)
        ]
        if intervals.size < 3:
            return -1
        median_interval = float(np.median(intervals))
        relative_deviation = np.abs(intervals - median_interval) / median_interval
        intervals = intervals[relative_deviation <= 0.35]
        if intervals.size < 3:
            return -1
        return float(60.0 / np.median(intervals))
    except Exception:
        return -1

