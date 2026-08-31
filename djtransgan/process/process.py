import torch

from djtransgan.config import settings
from djtransgan.utils import normalize, samples_to_time, squeeze_dim
from djtransgan.dataset import select_audio_region
from djtransgan.process import sync_bpm, sync_cue
from djtransgan.process import estimate_beat, select_cue_points, correct_cue


def preprocess(prev_audio,
               next_audio,
               prev_cue,
               next_cue,
               on_progress=None,
               *,
               match_bpm: bool = False,
               align_cue: bool = True):
    """
    Prepare a track pair for the generator.

    Paper defaults (when flags are on): stretch Next → Prev BPM, then align
    Next cue window length to Prev. Studio defaults match_bpm=False so mixes
    keep both tracks at native tempo unless the user opts in.
    """
    total = 5

    def _progress(step, message):
        print(message)
        if on_progress is not None:
            on_progress(step, total, message)

    # #region agent log
    _cue_in = {"prev_cue_in": float(prev_cue), "next_cue_in": float(next_cue),
               "match_bpm": bool(match_bpm), "align_cue": bool(align_cue)}
    # #endregion

    _progress(1, '[1/5] beat tracking start ...')
    _, prev_bpm, _, prev_downbeat = estimate_beat(prev_audio)
    _, next_bpm, _, next_downbeat = estimate_beat(next_audio)
    _progress(1, '[1/5] beat tracking complete ...')

    _progress(2, '[2/5] bpm matching start ...')
    if match_bpm:
        next_audio, ratio = sync_bpm(next_audio, prev_bpm, next_bpm)
        next_downbeat = next_downbeat / ratio
        next_cue = correct_cue(next_downbeat, next_cue / ratio)
    else:
        ratio = 1.0
        next_cue = correct_cue(next_downbeat, next_cue)
    prev_cue = correct_cue(prev_downbeat, prev_cue)
    # #region agent log
    try:
        import json, time
        from pathlib import Path
        _p = Path(__file__).resolve().parents[3] / "debug-3fef21.log"
        with open(_p, "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "3fef21",
                "hypothesisId": "H2-H3-H4",
                "location": "process/process.py:after_correct_cue",
                "message": "cues after snap / bpm",
                "data": {
                    **_cue_in,
                    "prev_cue_snap": float(prev_cue),
                    "next_cue_snap": float(next_cue),
                    "ratio": float(ratio),
                    "prev_bpm": float(prev_bpm) if prev_bpm is not None else None,
                    "next_bpm": float(next_bpm) if next_bpm is not None else None,
                    "next_downbeat_n": int(len(next_downbeat)),
                    "next_downbeat_first": float(next_downbeat[0]) if len(next_downbeat) else None,
                    "next_downbeat_last": float(next_downbeat[-1]) if len(next_downbeat) else None,
                    "next_audio_sec": float(next_audio.size(-1)) / float(settings.SR),
                },
                "timestamp": int(time.time() * 1000),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
    _progress(2, '[2/5] bpm matching complete ...' if match_bpm else '[2/5] bpm matching skipped ...')

    _progress(3, '[3/5] cue point select start ...')
    prev_cues, next_cues = select_cue_points(prev_cue, next_cue, prev_downbeat, next_downbeat)
    prev_cue, next_cue = float(prev_cues[1]), float(next_cues[1])
    _progress(3, '[3/5] cue point select complete ...')

    _progress(4, '[4/5] cue region alignment start ...')
    if align_cue:
        # #region agent log
        try:
            import json, time
            from pathlib import Path
            _p = Path(__file__).resolve().parents[3] / "debug-3fef21.log"
            with open(_p, "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "sessionId": "3fef21",
                    "hypothesisId": "H5",
                    "location": "process/process.py:before_sync_cue",
                    "message": "about to sync_cue",
                    "data": {
                        "prev_cues": [float(prev_cues[0]), float(prev_cues[1])],
                        "next_cues": [float(next_cues[0]), float(next_cues[1])],
                        "next_zero": abs(float(next_cues[1]) - float(next_cues[0])) < 1e-6,
                    },
                    "timestamp": int(time.time() * 1000),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion
        next_audio, next_cues = sync_cue(prev_audio, next_audio, prev_cues, next_cues)
        _progress(4, '[4/5] cue region alignment complete ...')
    else:
        _progress(4, '[4/5] cue region alignment skipped ...')

    _progress(5, '[5/5] normalize start ...')
    next_audio = normalize(next_audio)
    prev_audio = normalize(prev_audio)
    _progress(5, '[5/5] normalize complete ...')

    prev_audio_for_g, prev_cues_for_g, (prev_cues_ori, prev_timestamps) = select_audio_region(
        prev_audio, prev_cues, settings.N_TIME, True, 0)
    next_audio_for_g, next_cues_for_g, (next_cues_ori, next_timestamps) = select_audio_region(
        next_audio, next_cues, settings.N_TIME, True, 1)
    pair_audio = [prev_audio, next_audio]
    timestamps = [prev_timestamps, next_timestamps]
    pair_audio_for_g = [prev_audio_for_g.unsqueeze(0), next_audio_for_g.unsqueeze(0).to(torch.float32)]
    cue_for_g = prev_cues_for_g.unsqueeze(0).to(torch.float32)

    meta = {
        'prev_bpm': float(prev_bpm) if prev_bpm is not None else None,
        'next_bpm': float(next_bpm) if next_bpm is not None else None,
        'prev_cue': float(prev_cue),
        'next_cue': float(next_cue),
        'prev_cues': [float(x) for x in prev_cues],
        'next_cues': [float(x) for x in next_cues],
        'stretch_ratio': float(ratio),
        'match_bpm': bool(match_bpm),
        'align_cue': bool(align_cue),
    }

    return (pair_audio, timestamps), (pair_audio_for_g, cue_for_g), meta


def postprocess(mix_audio, pair_audio, timestamp, cue):
    if len(mix_audio.size()) > 2:
        mix_audio = squeeze_dim(mix_audio)

    if len(cue.size()) > 1:
        cue = squeeze_dim(cue)

    new_cue = samples_to_time(timestamp[0][0] + cue * settings.N_TIME * settings.SR)

    prev_audio = pair_audio[0][:, :timestamp[0][0]]
    next_audio = pair_audio[1][:, timestamp[1][1]:]

    return torch.cat((prev_audio, mix_audio, next_audio), axis=1), new_cue

