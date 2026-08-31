import torch
import random
from djtransgan.config import settings

random.seed(settings.RANDOM_SEED)

try:
    from acoustics.generator import noise as _acoustics_noise
except ImportError:  # optional: training-only; inference does not need it
    _acoustics_noise = None


def generate_noise(secs, color='white'):
    if _acoustics_noise is None:
        raise ImportError(
            "package 'acoustics' is required for noise generation "
            "(training only; skip for Mix Studio inference)"
        )
    return torch.from_numpy(_acoustics_noise(secs * settings.SR, color)).to(torch.float32)
