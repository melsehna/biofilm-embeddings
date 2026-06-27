#!/usr/bin/env python3
"""Headless DINOv2 extraction over an ALREADY-PROCESSED output root (embed-only).

Skips image processing entirely. Walks <output_root> for every plate's
processedImages/index.csv, gathers the wells whose `_processed.tif` resolves
(NAS-mirrored / relocated trees are handled by basename re-resolution against
the index dir — see embeddings.extractor.resolveProcessedPath), runs ONE GPU
sweep over all of them, and writes <output_root>/embeddings/cls_cache.pt
(resumable via per-batch checkpoints under embeddings/batches/).

This is the headless equivalent of the GUI Run tab's "Extract DINOv2 embeddings
(phase 2)" button: point it at a tree produced by biofilm-processing (or this
GUI's phase 1) and it embeds the existing processed stacks without reprocessing.

Defaults match the GUI's ExtractWorker so CLI output is interchangeable with
GUI output.

Usage:
    biofilm-embeddings-run /path/to/output_root
    biofilm-embeddings-run /path/to/output_root --dry-run        # list wells, no GPU
    biofilm-embeddings-run /path/to/output_root --model facebook/dinov2-giant \
        --image-size 518 --well-batch 4 --workers 3

Exit codes: 0 = success, 2 = bad args / no wells found, 1 = extraction failed.
"""
# Thread-limiting + headless Qt must be set before numpy / PySide6 / torch load.
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
import argparse


def buildParser():
    p = argparse.ArgumentParser(
        prog='biofilm-embeddings-run',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('output_root',
                   help='Root dir containing <plate>/processedImages/index.csv '
                        'trees (e.g. a biofilm-processing NAS mirror). The cache '
                        'is written to <output_root>/embeddings/cls_cache.pt.')
    p.add_argument('--model', default='facebook/dinov2-base',
                   help='HuggingFace DINOv2 model id (default: facebook/dinov2-base).')
    p.add_argument('--image-size', type=int, default=518,
                   help='Square resize fed to the ViT (default: 518).')
    p.add_argument('--grid-size', type=int, default=3,
                   help='Patch-pool grid (gridSize²); default 3.')
    p.add_argument('--well-batch', type=int, default=4,
                   help='Wells per GPU batch (default: 4).')
    p.add_argument('--workers', type=int, default=3,
                   help='DataLoader workers (default: 3).')
    p.add_argument('--prefetch', type=int, default=2,
                   help='DataLoader prefetch factor (default: 2).')
    p.add_argument('--no-cls', dest='extract_cls', action='store_false', default=True,
                   help='Skip CLS-token extraction.')
    p.add_argument('--no-patches', dest='extract_patches', action='store_false', default=True,
                   help='Skip pooled-patch extraction.')
    p.add_argument('--dry-run', action='store_true',
                   help='Collect + report wells/plates/mags/nFrames and exit, '
                        'without loading the model or touching the GPU.')
    p.add_argument('--limit', type=int, default=None,
                   help='Embed only the first N wells found — for a fast smoke '
                        'test of GPU/model/write before the full run.')
    p.add_argument('--cache-dir', dest='cache_dir', default=None,
                   help='Write embeddings/ here instead of into <output_root> '
                        '(e.g. local scratch for a test, or when the source tree '
                        'is read-only). Reads still come from the source tree.')
    p.add_argument('--n-frames', dest='n_frames', type=int, default=None,
                   help='Frames per stack to embed. Default: inferred from the '
                        'first stack. Wells with FEWER frames are skipped (logged '
                        'to excluded_short_wells.csv); wells with more are '
                        'truncated. Pin this explicitly to be robust to whichever '
                        'stack happens to be first.')
    return p


def _frameCount(path):
    """Frame count of a processed stack via TIFF page headers — no pixel read.

    Verified to equal the dataset's true frame count (saveStack writes one page
    per frame), at ~10x lower cost than a full load, so it's cheap enough to
    pre-scan every well before extraction.
    """
    import tifffile
    try:
        with tifffile.TiffFile(path) as tf:
            return len(tf.pages)
    except Exception:
        return None


def main(argv=None):
    args = buildParser().parse_args(argv)

    root = args.output_root
    if not os.path.isdir(root):
        print(f'ERROR: not a directory: {root}', file=sys.stderr)
        return 2

    # _collectProcessedRows / _peekFrameCount are plain functions in the GUI
    # module (gui/__init__ and gui/tabs/__init__ are empty, so importing under
    # offscreen Qt creates no QApplication). Reusing them keeps the path
    # resolution + frame-count logic single-source with the GUI Extract button.
    from biofilm_embeddings.gui.tabs.run import _collectProcessedRows, _peekFrameCount

    def log(m):
        print(m, flush=True)

    log(f'Scanning {root} for processed.tif via plate index files…')
    rows = _collectProcessedRows(root, logFn=log)
    if not rows:
        print('ERROR: no resolvable _processed.tif found under the output root. '
              'Is this a biofilm-processing output tree? (each plate needs '
              'processedImages/index.csv with a `processed` column)', file=sys.stderr)
        return 2

    # nFrames: explicit --n-frames, else inferred from the first stack. Inferring
    # is fragile (the first stack might be a short one), so --n-frames pins it.
    nFrames = args.n_frames if args.n_frames is not None else _peekFrameCount(rows[0]['processed'])
    plates = sorted({r.get('plate', '') for r in rows})
    mags = sorted({r.get('mag', '') for r in rows})
    objs = sorted({r.get('objective', '') for r in rows if r.get('objective')})
    log('')
    log(f'  wells:        {len(rows)}')
    log(f'  plates:       {len(plates)}')
    log(f'  magnification: {mags}  objective(s): {objs or "n/a"}')
    log(f'  nFrames: {nFrames}' + ('' if args.n_frames is not None else ' (inferred from first stack)'))
    if len(mags) > 1:
        log(f'  WARNING: multiple magnifications {mags} in one tree — embeddings '
            f'mix physical scales and are NOT comparable. Run one magnification '
            f'per output root.')

    # Pre-filter: drop wells with FEWER than nFrames frames (e.g. a plate
    # acquired with one fewer timepoint). The dataset would otherwise hard-crash
    # on the first short stack, mid-run. Page-header count, cheap. The kept set
    # all has >= nFrames (the dataset truncates the extras), so the embeddings
    # are frame-aligned. Excluded wells are recorded next to the cache.
    log(f'  checking frame counts across {len(rows)} wells…')
    kept, shortRows = [], []
    for r in rows:
        fc = _frameCount(r['processed'])
        if fc is not None and fc >= nFrames:
            kept.append(r)
        else:
            r = dict(r); r['_frames'] = '' if fc is None else fc
            shortRows.append(r)
    if shortRows:
        from collections import Counter
        byPlate = Counter(r.get('plate', '') for r in shortRows)
        log(f'  SKIPPING {len(shortRows)} wells with < {nFrames} frames (or unreadable):')
        for pl, n in sorted(byPlate.items()):
            log(f'    {n} wells — plate {pl}')
        if not args.dry_run:
            import csv as _csv
            embDir = os.path.join(args.cache_dir or root, 'embeddings')
            os.makedirs(embDir, exist_ok=True)
            exPath = os.path.join(embDir, 'excluded_short_wells.csv')
            with open(exPath, 'w', newline='') as f:
                w = _csv.writer(f)
                w.writerow(['plate', 'well', 'mag', 'frames', 'needed', 'processed'])
                for r in shortRows:
                    w.writerow([r.get('plate', ''), r.get('well', ''), r.get('mag', ''),
                                r.get('_frames', ''), nFrames, r.get('processed', '')])
            log(f'  excluded wells recorded: {exPath}')
    rows = kept
    if not rows:
        print(f'ERROR: no wells left with >= {nFrames} frames.', file=sys.stderr)
        return 2
    log(f'  embedding {len(rows)} wells at {nFrames} frames')

    # --limit: smoke-test on the first N wells (after the full report above, so
    # you still see the true totals).
    if args.limit is not None and args.limit < len(rows):
        log(f'  --limit {args.limit}: embedding only the first {args.limit} of '
            f'{len(rows)} wells (smoke test).')
        rows = rows[:args.limit]

    # --cache-dir: write embeddings/ somewhere other than the source tree (test
    # scratch, or a read-only source). Reads still use each row's resolved path.
    cacheRoot = args.cache_dir or root
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        log(f'  cache dir: {cacheRoot}/embeddings  (source tree: {root})')

    if args.dry_run:
        log('\n[dry-run] not loading model / not extracting. Re-run without '
            '--dry-run to embed.')
        return 0

    # GPU advisory (do not init CUDA here; torch.cuda.is_available is NVML-based).
    try:
        import torch
        if not torch.cuda.is_available():
            log('  WARNING: no CUDA device visible — extraction will run on CPU '
                f'(hours-to-days for {len(rows)} wells × {nFrames} frames).')
        else:
            log('  CUDA available.')
    except Exception as e:
        log(f'  (could not probe torch CUDA: {e})')

    from biofilm_embeddings.embeddings.extractor import extractAll

    def progress(done, total):
        print(f'  [extract] batch {done}/{max(total, 1)}', flush=True)

    cachePath = extractAll(
        rows, cacheRoot,
        modelName=args.model,
        imageSize=args.image_size,
        nFrames=nFrames,
        extractCls=args.extract_cls,
        extractPatches=args.extract_patches,
        gridSize=args.grid_size,
        wellBatch=args.well_batch,
        workers=args.workers,
        prefetch=args.prefetch,
        progressFn=log,
        batchProgressFn=progress,
    )

    if cachePath:
        log(f'\nCache written: {cachePath}')
        return 0
    print('ERROR: extraction did not produce a cache (cancelled or failed).',
          file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
