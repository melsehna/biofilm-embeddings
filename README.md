# µTyper-Vision

PySide6 GUI for biofilm phenotyping with frozen DINOv2 ViT embeddings.

Two phases, one window:

1. **Image processing.** Raw Cytation TIFFs become registered, segmented `_processed.tif` stacks. Lifted from `biofilm-processing` / `phenotypr`.
2. **DINOv2 ViT embedding extraction.** Frozen DINOv2-B/14 runs over every processed well, producing per-frame CLS and pooled patch tokens for downstream trajectory analysis (fPCA, UMAP, path signatures).

The two phases are separate buttons. Process now, extract later. Re-extract with a different model without reprocessing.

## Install

`pip install -e .` pulls everything (PySide6, opencv, scikit-image, tifffile, transformers, **torch>=2.7**) from PyPI. Python 3.10+ recommended.

```bash
cd ~/microTyper-Vision
pip install -e .
```

### GPU compatibility

PyTorch 2.7+ is pinned because that's the first stable release whose default PyPI wheels include Blackwell (sm_120) SASS kernels. Lower versions silently fail on RTX 50-series GPUs at the first forward pass.

| Hardware | Status | Notes |
|---|---|---|
| Pascal / Turing / Ampere / Hopper (x86_64) | works | Default `pip install -e .` resolves a torch wheel that includes your arch's kernels. |
| RTX 50-series Blackwell (x86_64) | works (with torch>=2.7, pinned) | First run downloads a cu126 wheel from PyPI; sm_120 kernels are included. |
| DGX Spark / Blackwell ARM64 (aarch64) | needs extra step | PyPI's aarch64 `torch` is CPU-only. Install CUDA torch first from PyTorch's cu128 index, then this package: `pip install torch --index-url https://download.pytorch.org/whl/cu128 && pip install -e .` |
| No GPU | works | Phase 2 silently falls back to CPU. Real biofilm runs (a few hundred wells x 31 frames at 518 px) take hours to days on CPU. Fine for a smoke test, not for production. |

A VRAM probe and auto-batch-adjust is not implemented. If you hit CUDA OOM on a smaller card, lower `extractionWellBatch` in the Parameters tab (try 2 or 1), drop `imageSize` to 364, or switch to `facebook/dinov2-small`.

### Dev env note

The sibling `~/embeddings/` repo runs in a conda env called `embeddings` with torch 2.4. That's too old for sm_120, and `pip install -e .` for this repo will try to upgrade it. If you want to keep the old env pinned and avoid disrupting `~/embeddings/`'s scripts, create a fresh env for µTyper-Vision instead.

## Run

```bash
mtv-gui
```

### Desktop shortcut (optional)

`scripts/installDesktopShortcut.py` installs a launcher in the OS's application menu (and on the Desktop if one exists) using `assets/dora5.jpg` as the icon. The launcher auto-activates the conda env you ran the installer from.

```bash
python scripts/installDesktopShortcut.py
```

Works on Linux (`.desktop`), macOS (`.app` bundle), and Windows (`.bat` + `.lnk`). On macOS and Windows, JPG is not a valid icon format for Finder / Explorer respectively, so for those platforms drop a converted icon next to `dora5.jpg`:

- macOS: `sips -s format icns assets/dora5.jpg --out assets/dora5.icns`
- Windows: `magick convert assets/dora5.jpg -define icon:auto-resize=256,128,64,48,32,16 assets/dora5.ico`

Re-run the installer after converting. To remove later, delete `microtyper-vision.desktop` from `~/.local/share/applications/` (Linux), `µTyper-Vision.app` from `~/Desktop/` (macOS), or `µTyper-Vision.lnk` from the Desktop (Windows).

## GUI tabs

| Tab | Purpose |
|---|---|
| Setup | Plate folder picker, output dir, magnification auto-detection from Cytation TIFF metadata |
| Parameters | Preprocessing knobs, per-magnification overrides, DINOv2 model/grid/batch settings |
| Preview | Live raw / normalized / mask view at the current parameters |
| Conditions | Per-plate well-condition assignment (6/12/24/48/96/384-well formats) |
| Test Well | Run both phases on a single well; preview raw, processed, and CLS PC1 trajectory |
| Run | Start processing (phase 1) and Extract DINOv2 embeddings (phase 2) |

## Output layout

```
<outputRoot>/
├── <plate>/
│   ├── processedImages/
│   │   ├── index.csv                            # per-well rows: plate, well, mag, paths, pxToUm, objective
│   │   ├── run_params.json                      # phase-1 resume key
│   │   ├── <wellId>_processed.tif               # (T, H, W) float32, [0, 1]
│   │   ├── <wellId>_registered_raw.tif
│   │   ├── <wellId>_masks.npz                   # key 'masks', bool
│   │   ├── <wellId>_biomass.csv
│   │   └── <wellId>_overlay.mp4
│   └── ...
└── embeddings/
    └── cls_cache.pt                             # consolidated DINOv2 cache
                                                  # keys: cls (W,T,D), patches (W,T,G²,D),
                                                  #       wells, plates, index, gridSize, model
```

## Project context

This repo bundles a copied subset of `~/biofilm-processing` (the per-well processing core only, no colony tracking, whole-image, or intensity feature extraction) with the DINOv2 + dataset code originally prototyped in `~/embeddings/`.
