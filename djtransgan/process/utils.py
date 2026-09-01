import numpy as np
from djtransgan.config import settings
from djtransgan.utils  import find_nearest, find_index, time_to_samples


def select_cue_points(prev_cue, next_cue, prev_downbeat, next_downbeat):
    # CUE_BAR bars before the cue; clamp so early cues don't wrap via negative index.
    prev_i = int(find_index(prev_downbeat, prev_cue))
    next_i = int(find_index(next_downbeat, next_cue))
    # Need CUE_BAR bars *before* the cue or the mix window length is 0.
    min_i = int(settings.CUE_BAR)
    if next_i < min_i and len(next_downbeat) > min_i:
        next_i = min_i
        next_cue = float(next_downbeat[next_i])
    if prev_i < min_i and len(prev_downbeat) > min_i:
        prev_i = min_i
        prev_cue = float(prev_downbeat[prev_i])
    prev_start = max(0, prev_i - settings.CUE_BAR)
    next_start = max(0, next_i - settings.CUE_BAR)
    # Last-resort: still empty (very short downbeat list) → use adjacent downbeats.
    if next_start >= next_i and len(next_downbeat) >= 2:
        next_i = min(1, len(next_downbeat) - 1)
        next_start = 0
        next_cue = float(next_downbeat[next_i])
    if prev_start >= prev_i and len(prev_downbeat) >= 2:
        prev_i = min(1, len(prev_downbeat) - 1)
        prev_start = 0
        prev_cue = float(prev_downbeat[prev_i])
    prev_cues = [prev_downbeat[prev_start], prev_cue]
    next_cues = [next_downbeat[next_start], next_cue]
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
