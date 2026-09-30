"""Resolve dataset font checks locally; never request a font download."""
from pathlib import Path

FONT = Path(__file__).resolve().parent/'assets/DejaVuLGCSans.ttf'


def offline_font():
    if not FONT.is_file():
        raise FileNotFoundError(f'Prepare the recorded local font first: {FONT}')
    from ultralytics.data import utils
    def resolve(font='Arial.ttf'):
        # Explicit DejaVu substitute, not a renamed Arial asset. Training plots
        # are disabled; this satisfies the unconditional dataset font check.
        return FONT
    utils.check_font = resolve
