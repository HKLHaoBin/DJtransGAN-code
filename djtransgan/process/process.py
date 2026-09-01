import torch

from djtransgan.config import settings
from djtransgan.utils import normalize, samples_to_time, squeeze_dim, time_to_samples
from djtransgan.dataset import select_audio_region
from djtransgan.process import estimate_beat, select_cue_points, correct_cue
from djtransgan.process.tempo import (
    MAX_TEMPO_RATE_DELTA,
    build_tempo_plan,
    build_transition_window,
    estimate_local_tempo,
    resolve_tempo_match,
)


POSTPROCESS_CROSSFADE_SECONDS = 0.03


def preprocess(prev_audio,
               next_audio,
               prev_cue,
               next_cue,
               on_progress=None,
               *,
               match_bpm: bool = False,
               align_cue: bool = True,
               max_tempo_rate_delta: float = MAX_TEMPO_RATE_DELTA):
    """Prepare original-speed tracks and one optional warped Next model window."""
    total = 5

    def _progress(step, message):
        print(message)
        if on_progress is not None:
            on_progress(step, total, message)

    _progress(1, '[1/5] beat tracking start ...')
    _, prev_bpm, prev_beats, prev_downbeat = estimate_beat(prev_audio)
    _, next_bpm, next_beats, next_downbeat = estimate_beat(next_audio)
    _progress(1, '[1/5] beat tracking complete ...')

    _progress(2, '[2/5] cue correction start ...')
    next_cue = correct_cue(next_downbeat, next_cue)
    prev_cue = correct_cue(prev_downbeat, prev_cue)
    _progress(2, '[2/5] cue correction complete ...')

    _progress(3, '[3/5] cue window and local tempo analysis start ...')
    prev_cues, next_cues = select_cue_points(prev_cue, next_cue, prev_downbeat, next_downbeat)
    prev_cue, next_cue = float(prev_cues[1]), float(next_cues[1])
    prev_local = estimate_local_tempo(prev_beats, prev_cue, fallback_bpm=prev_bpm)
    next_local = estimate_local_tempo(next_beats, next_cue, fallback_bpm=next_bpm)
    tempo_match = resolve_tempo_match(
        prev_local.bpm if prev_local.bpm is not None else float('nan'),
        next_local.bpm if next_local.bpm is not None else float('nan'),
    )
    _progress(3, '[3/5] cue window and local tempo analysis complete ...')

    _progress(4, '[4/5] normalize original-speed tracks start ...')
    next_audio = normalize(next_audio)
    prev_audio = normalize(prev_audio)
    _progress(4, '[4/5] normalize original-speed tracks complete ...')

    _progress(5, '[5/5] unified transition time-map preparation start ...')
    prev_audio_for_g, prev_cues_for_g, (prev_cues_ori, prev_timestamps) = select_audio_region(
        prev_audio, prev_cues, settings.N_TIME, True, 0)
    next_audio_for_g, next_cues_for_g, (next_cues_ori, next_timestamps) = select_audio_region(
        next_audio, next_cues, settings.N_TIME, True, 1)
    model_frames = int(round(float(settings.N_TIME) * float(settings.SR)))
    tempo_requested = bool(match_bpm or align_cue)
    transition = None
    if tempo_requested and tempo_match.normalized_rate is not None:
        cue_position = prev_cues_for_g[0] if align_cue else next_cues_for_g[0]
        target_start_frame = int(round(float(cue_position) * model_frames))
        source_start_frame = int(time_to_samples(next_cues[0]))
        hold_source_frames = (
            int(time_to_samples(float(next_cues[1]) - float(next_cues[0])))
            if align_cue
            else 0
        )
        transition = build_transition_window(
            next_audio,
            source_start_frame=source_start_frame,
            target_start_frame=target_start_frame,
            model_frames=model_frames,
            prev_bpm=prev_local.bpm,
            next_bpm=next_local.bpm,
            hold_source_frames=hold_source_frames,
            max_delta=max_tempo_rate_delta,
            sample_rate=settings.SR,
        )
        next_audio_for_g = transition.audio
        next_timestamps = [transition.source_start_frame, transition.source_end_frame]
        plan = transition.plan
    else:
        plan = build_tempo_plan(1.0, 1.0, int(next_audio_for_g.size(-1)))
    _progress(5, '[5/5] unified transition time-map preparation complete ...')

    pair_audio = [prev_audio, next_audio]
    timestamps = [prev_timestamps, next_timestamps]
    pair_audio_for_g = [prev_audio_for_g.unsqueeze(0), next_audio_for_g.unsqueeze(0).to(torch.float32)]
    cue_for_g = prev_cues_for_g.unsqueeze(0).to(torch.float32)

    fallback_reasons = []
    if prev_local.reason is not None:
        fallback_reasons.append(f"prev:{prev_local.reason}")
    if next_local.reason is not None:
        fallback_reasons.append(f"next:{next_local.reason}")
    if tempo_match.reason is not None:
        fallback_reasons.append(tempo_match.reason)
    tempo_disabled_reason = None
    if not tempo_requested:
        tempo_disabled_reason = 'tempo_controls_disabled'
    elif tempo_match.normalized_rate is None:
        tempo_disabled_reason = tempo_match.reason
    elif not plan.enabled:
        tempo_disabled_reason = 'native_rate_identity'

    source_start = transition.source_start_frame if transition is not None else int(next_timestamps[0])
    source_end = transition.source_end_frame if transition is not None else int(next_timestamps[1])
    target_start = transition.target_start_frame if transition is not None else 0
    target_end = transition.target_end_frame if transition is not None else model_frames
    keyframe_summary = [[int(source), int(target)] for source, target in plan.keyframes]
    rate_summary = [[int(source), float(rate)] for source, rate in plan.rate_keyframes]

    meta = {
        'prev_bpm': float(prev_bpm) if prev_bpm is not None else None,
        'next_bpm': float(next_bpm) if next_bpm is not None else None,
        'prev_local_bpm': float(prev_local.bpm) if prev_local.bpm is not None else None,
        'next_local_bpm': float(next_local.bpm) if next_local.bpm is not None else None,
        'prev_local_bpm_source': prev_local.source,
        'next_local_bpm_source': next_local.source,
        'prev_local_bpm_reason': prev_local.reason,
        'next_local_bpm_reason': next_local.reason,
        'prev_cue': float(prev_cue),
        'next_cue': float(next_cue),
        'prev_cues': [float(x) for x in prev_cues],
        'next_cues': [float(x) for x in next_cues],
        'stretch_ratio': float(plan.applied_start_rate),
        'match_bpm': bool(match_bpm),
        'align_cue': bool(align_cue),
        'raw_tempo_rate': tempo_match.raw_rate,
        'normalized_tempo_rate': tempo_match.normalized_rate,
        'applied_start_rate': float(plan.applied_start_rate),
        'max_delta': float(plan.max_delta),
        'tempo_rate_clamped': bool(plan.clamped),
        'tempo_transition_applied': bool(transition is not None and plan.enabled),
        'tempo_fallback_reason': ';'.join(fallback_reasons) if fallback_reasons else None,
        'tempo_disabled_reason': tempo_disabled_reason,
        'tempo_curve': {
            'shape': 'hold_then_smoothstep',
            'keyframes': keyframe_summary,
            'rate_keyframes': rate_summary,
            'end_rate': float(plan.rate_keyframes[-1][1]),
            'fitted_target_frames': (
                int(transition.fitted_target_frames) if transition is not None else 0
            ),
        },
        'tempo_source_anchor': {
            'start_frame': source_start,
            'end_frame': source_end,
        },
        'tempo_target_anchor': {
            'start_frame': target_start,
            'end_frame': target_end,
        },
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

    requested_crossfade = int(round(POSTPROCESS_CROSSFADE_SECONDS * settings.SR))
    crossfade_frames = min(requested_crossfade, mix_audio.size(-1), next_audio.size(-1))
    if crossfade_frames > 0:
        progress = torch.linspace(
            0.0,
            1.0,
            crossfade_frames + 2,
            dtype=mix_audio.dtype,
            device=mix_audio.device,
        )[1:-1]
        phase = progress * (torch.pi / 2.0)
        left_weight = torch.cos(phase).unsqueeze(0)
        right_weight = torch.sin(phase).unsqueeze(0)
        blend = (
            mix_audio[:, -crossfade_frames:] * left_weight
            + next_audio[:, :crossfade_frames] * right_weight
        )
        transition_and_suffix = torch.cat(
            (
                mix_audio[:, :-crossfade_frames],
                blend,
                next_audio[:, crossfade_frames:],
            ),
            dim=1,
        )
    else:
        transition_and_suffix = torch.cat((mix_audio, next_audio), dim=1)

    return torch.cat((prev_audio, transition_and_suffix), axis=1), new_cue

