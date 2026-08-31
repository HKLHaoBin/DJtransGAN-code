import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
from pathlib import Path

# Allow `python script/inference.py` from code/, and also from workspace root.
_CODE = Path(__file__).resolve().parent.parent
_ROOT = _CODE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from server.engine import run_mix
from server.paths import ensure_runtime_env


def main():
    ensure_runtime_env()
    os.chdir(_CODE)

    from djtransgan.config import settings

    parser = argparse.ArgumentParser(description='DJtransGAN inference')
    parser.add_argument('--out_dir', type=str, default=os.path.join(settings.STORE_DIR, 'inference'))
    parser.add_argument('--g_path', type=str, default='./pretrained/djtransgan_minmax.pt')
    parser.add_argument('--prev_track', type=str, default='./test/Breikthru ft Danny Devinci-Touch.mp3')
    parser.add_argument('--next_track', type=str, default='./test/Jameson-Hangin.mp3')
    parser.add_argument('--prev_cue', type=float, default=96.0)
    parser.add_argument('--next_cue', type=float, default=30.0)
    parser.add_argument('--download', type=int, default=1)
    parser.add_argument('--match_bpm', type=int, default=1, help='1=stretch Next to Prev BPM')
    parser.add_argument('--align_cue', type=int, default=1, help='1=align cue-window lengths')

    args = parser.parse_args()

    def on_progress(step, total, message):
        print(message)

    from djtransgan.utils import get_filename

    saved_id = f'{get_filename(args.prev_track)}_{get_filename(args.next_track)}'
    result = run_mix(
        args.prev_track,
        args.next_track,
        args.prev_cue,
        args.next_cue,
        args.out_dir,
        weights_path=Path(args.g_path) if args.g_path else None,
        download=bool(args.download),
        on_progress=on_progress,
        short_name=f'{saved_id}_short.wav',
        full_name=f'{saved_id}_full.wav',
        params_name=f'{saved_id}_params.json',
        match_bpm=bool(args.match_bpm),
        align_cue=bool(args.align_cue),
    )
    print('Wrote:', result['short_path'])
    print('Wrote:', result['full_path'])
    print('Wrote:', result['params_path'])


if __name__ == '__main__':
    main()
