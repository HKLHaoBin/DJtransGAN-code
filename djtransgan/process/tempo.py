from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
import pyrubberband as pyrb
import torch


MIN_PLAUSIBLE_BPM = 40.0
MAX_PLAUSIBLE_BPM = 240.0
MIN_LOCAL_INTERVALS = 3
MAX_INTERVAL_DEVIATION_RATIO = 0.35
MAX_TEMPO_RATE_DELTA = 0.08
DEFAULT_TIMEMAP_KEYFRAME_COUNT = 65
MAX_TEMPO_HOLD_FRACTION = 0.75


@dataclass(frozen=True)
class TempoEstimate:
    bpm: Optional[float]
    source: str
    reason: Optional[str]
    intervals_used: int


@dataclass(frozen=True)
class TempoMatch:
    raw_rate: Optional[float]
    normalized_rate: Optional[float]
    prev_multiplier: float
    next_multiplier: float
    reason: Optional[str]


@dataclass(frozen=True)
class TempoPlan:
    enabled: bool
    raw_rate: Optional[float]
    normalized_rate: Optional[float]
    applied_start_rate: float
    max_delta: float
    clamped: bool
    keyframes: tuple
    rate_keyframes: tuple
    reason: Optional[str]


@dataclass(frozen=True)
class TransitionWindow:
    audio: torch.Tensor
    plan: TempoPlan
    source_start_frame: int
    source_end_frame: int
    target_start_frame: int
    target_end_frame: int
    padded_source_frames: int
    fitted_target_frames: int


TEMPO_OCTAVE_MULTIPLIERS = (0.5, 1.0, 2.0)


def resolve_tempo_match(prev_bpm: float, next_bpm: float) -> TempoMatch:
    """Choose the half/double-tempo interpretation requiring the least speed change."""
    prev = float(prev_bpm)
    next_ = float(next_bpm)
    if not math.isfinite(prev) or not math.isfinite(next_) or prev <= 0.0 or next_ <= 0.0:
        return TempoMatch(
            raw_rate=None,
            normalized_rate=None,
            prev_multiplier=1.0,
            next_multiplier=1.0,
            reason="invalid_local_bpm",
        )
    raw_rate = prev / next_
    candidates = []
    for prev_multiplier in TEMPO_OCTAVE_MULTIPLIERS:
        for next_multiplier in TEMPO_OCTAVE_MULTIPLIERS:
            rate = (prev * prev_multiplier) / (next_ * next_multiplier)
            octave_changes = abs(math.log2(prev_multiplier)) + abs(math.log2(next_multiplier))
            candidates.append(
                (abs(math.log(rate)), octave_changes, rate, prev_multiplier, next_multiplier)
            )
    _, _, rate, prev_multiplier, next_multiplier = min(candidates)
    return TempoMatch(
        raw_rate=raw_rate,
        normalized_rate=rate,
        prev_multiplier=prev_multiplier,
        next_multiplier=next_multiplier,
        reason=None,
    )


def build_tempo_plan(
    prev_bpm: float,
    next_bpm: float,
    source_frames: int,
    *,
    max_delta: float = MAX_TEMPO_RATE_DELTA,
    hold_source_frames: int = 0,
) -> TempoPlan:
    """Build a smooth source-to-target map whose local tempo returns to 1.0."""
    max_delta = float(max_delta)
    if not math.isfinite(max_delta) or max_delta < 0.0 or max_delta >= 1.0:
        raise ValueError("max_delta must be finite and in [0, 1)")
    match = resolve_tempo_match(prev_bpm, next_bpm)
    source_frames = int(source_frames)
    if source_frames <= 0:
        return TempoPlan(
            enabled=False,
            raw_rate=match.raw_rate,
            normalized_rate=match.normalized_rate,
            applied_start_rate=1.0,
            max_delta=max_delta,
            clamped=False,
            keyframes=((0, 0),),
            rate_keyframes=((0, 1.0),),
            reason="empty_transition",
        )
    if match.normalized_rate is None:
        identity = ((0, 0), (source_frames, source_frames))
        identity_rates = ((0, 1.0), (source_frames, 1.0))
        return TempoPlan(
            enabled=False,
            raw_rate=match.raw_rate,
            normalized_rate=match.normalized_rate,
            applied_start_rate=1.0,
            max_delta=max_delta,
            clamped=False,
            keyframes=identity,
            rate_keyframes=identity_rates,
            reason=match.reason,
        )
    lower_rate = 1.0 - max_delta
    upper_rate = 1.0 + max_delta
    applied_rate = min(max(float(match.normalized_rate), lower_rate), upper_rate)
    clamped = not math.isclose(applied_rate, float(match.normalized_rate))

    point_count = min(DEFAULT_TIMEMAP_KEYFRAME_COUNT, source_frames + 1)
    source_points = np.unique(
        np.linspace(0, source_frames, num=point_count, dtype=np.int64)
    )
    max_hold_frames = int(source_frames * MAX_TEMPO_HOLD_FRACTION)
    hold = min(max(int(hold_source_frames), 0), max_hold_frames)
    rates = []
    for source in source_points:
        if source <= hold:
            rate = applied_rate
        else:
            progress = (float(source) - hold) / (source_frames - hold)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            rate = applied_rate + (1.0 - applied_rate) * smooth
        rates.append(float(rate))

    target_float = [0.0]
    for index in range(1, len(source_points)):
        source_delta = int(source_points[index] - source_points[index - 1])
        inverse_rate = 0.5 * ((1.0 / rates[index - 1]) + (1.0 / rates[index]))
        target_float.append(target_float[-1] + source_delta * inverse_rate)

    target_points = [0]
    for target in target_float[1:]:
        target_points.append(max(target_points[-1] + 1, int(round(target))))

    keyframes = tuple(
        (int(source), int(target))
        for source, target in zip(source_points.tolist(), target_points)
    )
    rate_keyframes = tuple(
        (int(source), float(rate))
        for source, rate in zip(source_points.tolist(), rates)
    )
    return TempoPlan(
        enabled=not math.isclose(applied_rate, 1.0),
        raw_rate=match.raw_rate,
        normalized_rate=match.normalized_rate,
        applied_start_rate=applied_rate,
        max_delta=max_delta,
        clamped=clamped,
        keyframes=keyframes,
        rate_keyframes=rate_keyframes,
        reason=match.reason,
    )


def apply_tempo_plan(audio, plan: TempoPlan, *, sample_rate: int):
    """Apply one tempo plan without mutating or aliasing the caller's audio."""
    if not plan.enabled:
        return audio.clone() if isinstance(audio, torch.Tensor) else np.array(audio, copy=True)

    is_tensor = isinstance(audio, torch.Tensor)
    if is_tensor:
        original_dtype = audio.dtype
        original_device = audio.device
        tensor_audio = audio.detach().cpu()
        if tensor_audio.ndim == 1:
            samples = tensor_audio.numpy()
            layout = "mono"
        elif tensor_audio.ndim == 2:
            samples = tensor_audio.transpose(0, 1).contiguous().numpy()
            layout = "channels"
        elif tensor_audio.ndim == 3 and tensor_audio.size(0) == 1:
            samples = tensor_audio[0].transpose(0, 1).contiguous().numpy()
            layout = "batch_channels"
        else:
            raise ValueError(
                "tempo audio tensor must be [frames], [channels, frames], "
                "or [1, channels, frames]"
            )
    else:
        samples = np.asarray(audio)
        layout = "numpy"

    if len(samples) != plan.keyframes[-1][0]:
        raise ValueError(
            f"tempo plan source length {plan.keyframes[-1][0]} "
            f"does not match audio length {len(samples)}"
        )

    stretched = pyrb.timemap_stretch(
        samples,
        int(sample_rate),
        list(plan.keyframes),
        rbargs={"--fine": ""},
    )
    if not is_tensor:
        return stretched

    output = torch.from_numpy(np.asarray(stretched))
    if layout == "channels":
        output = output.transpose(0, 1).contiguous()
    elif layout == "batch_channels":
        output = output.transpose(0, 1).contiguous().unsqueeze(0)
    return output.to(device=original_device, dtype=original_dtype)


def source_to_target_frame(source_frame: float, keyframes: Sequence[tuple]) -> float:
    """Map a source frame into warped target coordinates with endpoint clamping."""
    sources = np.asarray([point[0] for point in keyframes], dtype=np.float64)
    targets = np.asarray([point[1] for point in keyframes], dtype=np.float64)
    return float(np.interp(float(source_frame), sources, targets))


def target_to_source_frame(target_frame: float, keyframes: Sequence[tuple]) -> float:
    """Invert a strictly monotonic timemap with endpoint clamping."""
    sources = np.asarray([point[0] for point in keyframes], dtype=np.float64)
    targets = np.asarray([point[1] for point in keyframes], dtype=np.float64)
    return float(np.interp(float(target_frame), targets, sources))


def _solve_source_frames(
    target_frames: int,
    prev_bpm: float,
    next_bpm: float,
    max_delta: float,
    hold_source_frames: int,
) -> int:
    """Find the source span whose integrated map is closest to target_frames."""
    target_frames = int(target_frames)
    if target_frames <= 0:
        return 0

    def mapped_length(source_frames: int) -> int:
        plan = build_tempo_plan(
            prev_bpm,
            next_bpm,
            source_frames,
            max_delta=max_delta,
            hold_source_frames=hold_source_frames,
        )
        return int(plan.keyframes[-1][1])

    low = 1
    high = max(target_frames, 1)
    while mapped_length(high) < target_frames:
        high *= 2

    best_source = low
    best_error = abs(mapped_length(low) - target_frames)
    while low <= high:
        middle = (low + high) // 2
        error = mapped_length(middle) - target_frames
        absolute_error = abs(error)
        if absolute_error < best_error:
            best_source = middle
            best_error = absolute_error
        if error < 0:
            low = middle + 1
        elif error > 0:
            high = middle - 1
        else:
            return middle
    return best_source


def _pad_tensor_end(audio: torch.Tensor, frames: int) -> torch.Tensor:
    if frames <= 0:
        return audio
    shape = list(audio.shape)
    shape[-1] = int(frames)
    padding = torch.zeros(shape, dtype=audio.dtype, device=audio.device)
    return torch.cat((audio, padding), dim=-1)


def build_transition_window(
    audio: torch.Tensor,
    *,
    source_start_frame: int,
    target_start_frame: int,
    model_frames: int,
    prev_bpm: float,
    next_bpm: float,
    hold_source_frames: int = 0,
    max_delta: float = MAX_TEMPO_RATE_DELTA,
    sample_rate: int,
) -> TransitionWindow:
    """Warp only the Next transition source and anchor it in a fixed model window."""
    if not isinstance(audio, torch.Tensor) or audio.ndim < 1:
        raise ValueError("transition audio must be a torch tensor with a frame axis")
    model_frames = int(model_frames)
    source_start = min(max(int(source_start_frame), 0), int(audio.size(-1)))
    target_start = min(max(int(target_start_frame), 0), model_frames)
    active_target_frames = model_frames - target_start
    source_frames = _solve_source_frames(
        active_target_frames,
        prev_bpm,
        next_bpm,
        max_delta,
        hold_source_frames,
    )
    source_end_requested = source_start + source_frames
    source_end = min(source_end_requested, int(audio.size(-1)))
    source_audio = audio[..., source_start:source_end].clone()
    padded_source_frames = source_end_requested - source_end
    source_audio = _pad_tensor_end(source_audio, padded_source_frames)

    plan = build_tempo_plan(
        prev_bpm,
        next_bpm,
        source_frames,
        max_delta=max_delta,
        hold_source_frames=hold_source_frames,
    )
    warped = apply_tempo_plan(source_audio, plan, sample_rate=sample_rate)
    fitted_target_frames = active_target_frames - int(warped.size(-1))
    if fitted_target_frames > 0:
        warped = _pad_tensor_end(warped, fitted_target_frames)
    elif fitted_target_frames < 0:
        warped = warped[..., :active_target_frames]

    prefix_shape = list(audio.shape)
    prefix_shape[-1] = target_start
    prefix = torch.zeros(prefix_shape, dtype=audio.dtype, device=audio.device)
    model_audio = torch.cat((prefix, warped), dim=-1)
    return TransitionWindow(
        audio=model_audio,
        plan=plan,
        source_start_frame=source_start,
        source_end_frame=source_end,
        target_start_frame=target_start,
        target_end_frame=model_frames,
        padded_source_frames=padded_source_frames,
        fitted_target_frames=fitted_target_frames,
    )


def estimate_local_tempo(
    beat_timestamps: Sequence[float],
    cue_seconds: float,
    *,
    window_seconds: float = 16.0,
    fallback_bpm: Optional[float] = None,
) -> TempoEstimate:
    """Estimate tempo from beat intervals in a symmetric window around a cue."""
    beats = np.asarray(beat_timestamps, dtype=np.float64).reshape(-1)
    half_window = float(window_seconds) / 2.0
    local = beats[(beats >= cue_seconds - half_window) & (beats <= cue_seconds + half_window)]
    intervals = np.diff(local)
    min_interval = 60.0 / MAX_PLAUSIBLE_BPM
    max_interval = 60.0 / MIN_PLAUSIBLE_BPM
    intervals = intervals[
        np.isfinite(intervals)
        & (intervals >= min_interval)
        & (intervals <= max_interval)
    ]
    if intervals.size:
        median_interval = float(np.median(intervals))
        relative_deviation = np.abs(intervals - median_interval) / median_interval
        intervals = intervals[relative_deviation <= MAX_INTERVAL_DEVIATION_RATIO]
    if intervals.size < MIN_LOCAL_INTERVALS:
        fallback_valid = (
            fallback_bpm is not None
            and np.isfinite(float(fallback_bpm))
            and float(fallback_bpm) > 0.0
        )
        return TempoEstimate(
            bpm=float(fallback_bpm) if fallback_valid else None,
            source="fallback" if fallback_valid else "unavailable",
            reason=(
                "insufficient_local_intervals"
                if fallback_bpm is None or fallback_valid
                else "invalid_fallback_bpm"
            ),
            intervals_used=int(intervals.size),
        )
    bpm = 60.0 / float(np.median(intervals))
    return TempoEstimate(
        bpm=bpm,
        source="cue_local",
        reason=None,
        intervals_used=int(intervals.size),
    )
