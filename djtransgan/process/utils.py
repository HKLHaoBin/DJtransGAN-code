import numpy as np
from djtransgan.config import settings
from djtransgan.utils  import find_nearest, find_index, time_to_samples


def select_cue_points(prev_cue, next_cue, prev_downbeat, next_downbeat):
    # CUE_BAR bars before the cue; clamp so early cues don't wrap via negative index.
    prev_i = int(find_index(prev_downbeat, prev_cue))
    next_i = int(find_index(next_downbeat, next_cue))
    # #region agent log
    _bump = {"prev_bumped": False, "next_bumped": False, "prev_i_before": prev_i, "next_i_before": next_i}
    # #endregion
    # Need CUE_BAR bars *before* the cue or the mix window length is 0.
    min_i = int(settings.CUE_BAR)
    if next_i < min_i and len(next_downbeat) > min_i:
        next_i = min_i
        next_cue = float(next_downbeat[next_i])
        _bump["next_bumped"] = True
        _bump["next_i_after"] = next_i
        _bump["next_cue_after"] = float(next_cue)
    if prev_i < min_i and len(prev_downbeat) > min_i:
        prev_i = min_i
        prev_cue = float(prev_downbeat[prev_i])
        _bump["prev_bumped"] = True
        _bump["prev_i_after"] = prev_i
        _bump["prev_cue_after"] = float(prev_cue)
    prev_start = max(0, prev_i - settings.CUE_BAR)
    next_start = max(0, next_i - settings.CUE_BAR)
    # Last-resort: still empty (very short downbeat list) → use adjacent downbeats.
    if next_start >= next_i and len(next_downbeat) >= 2:
        next_i = min(1, len(next_downbeat) - 1)
        next_start = 0
        next_cue = float(next_downbeat[next_i])
        _bump["next_fallback_adjacent"] = True
    if prev_start >= prev_i and len(prev_downbeat) >= 2:
        prev_i = min(1, len(prev_downbeat) - 1)
        prev_start = 0
        prev_cue = float(prev_downbeat[prev_i])
        _bump["prev_fallback_adjacent"] = True
    prev_cues = [prev_downbeat[prev_start], prev_cue]
    next_cues = [next_downbeat[next_start], next_cue]
    # #region agent log
    try:
        import json, time
        from pathlib import Path
        _p = Path(__file__).resolve().parents[3] / "debug-3fef21.log"
        with open(_p, "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "3fef21",
                "runId": "post-fix",
                "hypothesisId": "H1",
                "location": "process/utils.py:select_cue_points",
                "message": "cue window indices",
                "data": {
                    "prev_i": prev_i,
                    "next_i": next_i,
                    "prev_start": prev_start,
                    "next_start": next_start,
                    "cue_bar": int(settings.CUE_BAR),
                    "next_clamped": _bump["next_i_before"] < settings.CUE_BAR,
                    "prev_cues": [float(prev_cues[0]), float(prev_cues[1])],
                    "next_cues": [float(next_cues[0]), float(next_cues[1])],
                    "next_len": float(next_cues[1]) - float(next_cues[0]),
                    "next_downbeat_n": int(len(next_downbeat)),
                    "next_downbeat_head": [float(x) for x in list(next_downbeat[: min(12, len(next_downbeat))])],
                    "bump": _bump,
                },
                "timestamp": int(time.time() * 1000),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
    return prev_cues, next_cues


def correct_cue(downbeat, cue):
    return downbeat[find_nearest(downbeat, cue)]


def filter_beat(beat, downbeat, sig_d):
    begin = np.where(beat == downbeat[0])[0][0]
    end = np.where(beat == downbeat[-1])[0][0] + sig_d
    return beat[begin:end]


def split_audio(audio, cue):
    cue_sample = [time_to_samples(c) for c in cue]
    before = None if cue_sample[0] == 0 else audio[:, :cue_sample[0]]
    middle = audio[:, cue_sample[0]:cue_sample[1]]
    after = None if cue_sample[1] == 0 else audio[:, cue_sample[1]:]
    return [before, middle, after]
