#!/usr/bin/env python3
"""Rebuild platform icons (.png, .ico, .icns) from assets/dora5.jpg.

Run this whenever the source image changes. The installer scripts look for
each format under its canonical name (`dora5.png` for Linux, `dora5.ico`
for Windows, `dora5.icns` for macOS).

Usage:
    python scripts/buildIcons.py
    python scripts/buildIcons.py path/to/other-source.jpg
"""

import sys
from pathlib import Path

from PIL import Image


ICO_SIZES = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
PNG_SIZE = (512, 512)


def buildIcons(srcPath):
    src = Path(srcPath).resolve()
    if not src.exists():
        sys.exit(f'source image not found: {src}')

    img = Image.open(src).convert('RGBA')
    print(f'source: {src}  ({img.size}, {img.mode})')

    # Square-crop from center so small-size variants don't get stretched.
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    print(f'square-cropped to {img.size}')

    pngPath = src.with_suffix('.png')
    img.resize(PNG_SIZE, Image.LANCZOS).save(pngPath)
    print(f'wrote {pngPath}  ({pngPath.stat().st_size:,} bytes)')

    icoPath = src.with_suffix('.ico')
    img.save(icoPath, format='ICO', sizes=ICO_SIZES)
    print(f'wrote {icoPath}  ({icoPath.stat().st_size:,} bytes, sizes {ICO_SIZES})')

    icnsPath = src.with_suffix('.icns')
    img.save(icnsPath, format='ICNS')
    print(f'wrote {icnsPath}  ({icnsPath.stat().st_size:,} bytes)')


def main():
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = Path(__file__).resolve().parent.parent / 'assets' / 'dora5.jpg'
    buildIcons(src)


if __name__ == '__main__':
    main()
