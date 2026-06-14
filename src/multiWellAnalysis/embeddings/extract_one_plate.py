"""CLI entry point: run DINOv2 extraction on a single plate's wells.

Designed to be invoked as a subprocess from ProcessingWorker (the GUI's
phase-1 worker) when per-plate pipelined mode is active. Running extraction
in a subprocess keeps CUDA out of the GUI parent process, which is important
because phase 1's ProcessPoolExecutor uses fork — and fork-after-CUDA-init
leaves forked workers with broken CUDA state.

Reads <plate_dir>/processedImages/index.csv and the `_processed.tif` it points
to (the single fixed-fpMean render in biofilm-processing >= v0.5.0), and writes
the resulting cache to <plate_dir>/embeddings/cls_cache.pt.

Exit codes: 0 = success, 2 = no usable wells found, anything else = failure.
"""

import argparse
import os
import sys


def _collectPlateRows(plateDir):
    """Walk a single plate's processedImages dir for its index.csv.

    Returns row dicts for extractAll. 'processed' = `_processed.tif`, which is the
    single fixed-fpMean render (biofilm-processing >= v0.5.0). The old
    `_processed_fpHalf.tif` swap is gone — that file no longer exists and
    `_processed.tif` is no longer the adaptive render.
    """
    import pandas as pd

    processedDir = os.path.join(plateDir, 'processedImages')
    indexPath = os.path.join(processedDir, 'index.csv')
    if not os.path.exists(indexPath):
        # also try the plateDir itself, in case the caller passes
        # processedImages directly
        indexPath = os.path.join(plateDir, 'index.csv')
        processedDir = plateDir
    if not os.path.exists(indexPath):
        raise FileNotFoundError(f'no index.csv under {plateDir}')

    df = pd.read_csv(indexPath, dtype=str).fillna('')
    if 'processed' not in df.columns:
        raise ValueError(f'{indexPath} has no `processed` column')
    df = df[df['processed'].apply(lambda p: bool(p) and os.path.exists(p))]

    rows = [row.to_dict() for _, row in df.iterrows()]
    print(f'  plate {os.path.basename(plateDir)}: {len(rows)} wells', flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plate-dir', required=True,
                        help='Path to the plate directory (parent of processedImages/)')
    parser.add_argument('--model', default='facebook/dinov2-base')
    parser.add_argument('--image-size', type=int, default=518)
    parser.add_argument('--n-frames', type=int, required=True,
                        help='Number of frames per stack')
    parser.add_argument('--extract-cls', action='store_true', default=True)
    parser.add_argument('--no-extract-cls', dest='extract_cls', action='store_false')
    parser.add_argument('--extract-patches', action='store_true', default=True)
    parser.add_argument('--no-extract-patches', dest='extract_patches', action='store_false')
    parser.add_argument('--grid-size', type=int, default=3)
    parser.add_argument('--well-batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--prefetch', type=int, default=2)
    args = parser.parse_args(argv)

    plateDir = os.path.abspath(args.plate_dir)
    if not os.path.isdir(plateDir):
        print(f'ERROR: plate dir does not exist: {plateDir}', file=sys.stderr)
        return 1

    rows = _collectPlateRows(plateDir)
    if not rows:
        print(f'  no usable wells under {plateDir} — nothing to extract', flush=True)
        return 2

    # Import torch + extractor lazily so --help is fast and so CUDA only
    # initializes inside this subprocess.
    from multiWellAnalysis.embeddings.extractor import extractAll

    def _log(msg):
        print(msg, flush=True)

    cachePath = extractAll(
        rows, plateDir,
        modelName=args.model,
        imageSize=args.image_size,
        nFrames=args.n_frames,
        extractCls=args.extract_cls,
        extractPatches=args.extract_patches,
        gridSize=args.grid_size,
        wellBatch=args.well_batch,
        workers=args.workers,
        prefetch=args.prefetch,
        progressFn=_log,
        batchProgressFn=lambda done, total: None,
        stopEvent=None,
    )

    if cachePath is None:
        print('extraction returned None (cancelled or failed)', file=sys.stderr)
        return 1
    print(f'cache written: {cachePath}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
