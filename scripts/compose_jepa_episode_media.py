"""Join the two preselected, verified model episodes for the README."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

from render_qualitative_rollouts import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renders', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    sources = []
    for directory in args.renders:
        receipt = json.loads((directory/'render_receipt.json').read_text())
        video = directory/'jepa_episode_hd.mp4'
        if sha(video) != receipt['outputs'][video.name]:
            raise ValueError('Rendered video differs from verification receipt')
        if receipt['all_states_checked'] != 100 or receipt['state_max_abs_error'] > 1e-6:
            raise ValueError('Incomplete state verification')
        sources.append({'video_sha256': sha(video), 'render': receipt})
    args.output.mkdir(parents=True)
    mp4 = args.output/'jepa_full_episodes_hd.mp4'
    gif = args.output/'jepa_full_episodes.gif'
    with tempfile.TemporaryDirectory() as temporary:
        manifest = Path(temporary)/'clips.txt'
        paths = [(directory/'jepa_episode_hd.mp4').resolve().as_posix()
                 for directory in args.renders]
        if any("'" in path or '\n' in path for path in paths):
            raise ValueError('Unsupported concat path')
        manifest.write_text(''.join(f"file '{path}'\n" for path in paths))
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'concat', '-safe', '0',
                        '-i', str(manifest), '-c', 'copy', '-movflags', '+faststart',
                        str(mp4)], check=True)
    # Fixed ordered dithering keeps the GIF compact without lowering resolution.
    filters = ('[0:v]fps=40,scale=1280:-1:flags=lanczos,split[a][b];'
               '[a]palettegen=stats_mode=diff[p];'
               '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(mp4), '-filter_complex',
                    filters, '-loop', '0', str(gif)], check=True)
    with Image.open(gif) as image:
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(image.info['duration'])
        resolution = list(image.size)
    receipt = {'role': 'two_preselected_qualitative_frozen_model_episodes',
               'order': ['reach', 'reach-wall'], 'sources': sources,
               'composer_sha256': sha(__file__),
               'mp4_resolution': [1920, 1080], 'gif_resolution': resolution,
               'gif_duration_seconds': sum(durations)/1000,
               'playback': 'simulation speed, omitting planner computation; 0.5 second hold per task',
               'outputs': {p.name: sha(p) for p in [mp4, gif]}}
    (args.output/'jepa_full_episodes_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
