# biofilm-embeddings

PySide6 GUI for biofilm phenotyping with frozen DINOv2 ViT embeddings.

Two phases, one window:

1. **Image processing.** Raw Cytation TIFFs become registered, segmented `_processed.tif` stacks. The processing engine is **not** a copy — it's `biofilm-processing` itself, vendored as a pinned git submodule (`external/biofilm-processing` @ v0.5.0) so the render is byte-for-byte identical and never drifts under the embeddings.
2. **DINOv2 ViT embedding extraction.** Frozen DINOv2-B/14 runs over every processed well, producing per-frame CLS and pooled patch tokens for downstream trajectory analysis (fPCA, UMAP, path signatures).

The two phases are separate buttons. Process now, extract later. Re-extract with a different model without reprocessing.

## Install

The processing engine (`biofilm-processing`) is vendored as a pinned git submodule and is
**not on PyPI**, so it must be fetched and installed editable **before** this package
(which pulls PySide6, opencv, scikit-image, tifffile, transformers, **torch>=2.7** from
PyPI). Python 3.10+ recommended.

**One command** (handles the submodule + install order for you):

```bash
git clone --recurse-submodules git@github.com:melsehna/biofilm-embeddings.git
cd biofilm-embeddings
bash scripts/setup.sh
```

Or the same steps by hand (the order matters):

```bash
git clone --recurse-submodules git@github.com:melsehna/biofilm-embeddings.git
cd biofilm-embeddings
git submodule update --init --recursive      # populate external/biofilm-processing @ v0.5.0
pip install -e external/biofilm-processing   # the pinned engine, FIRST
pip install -e .                             # then this package
```

The submodule is pinned (`biofilm-processing==0.5.0`) on purpose: the render is frozen at
that tag so embeddings stay comparable across batches. To upgrade deliberately, advance the
submodule to a newer tag and bump the `==` pin in `pyproject.toml` in the same commit.

### Install troubleshooting

- **`ERROR: No matching distribution found for biofilm-processing==0.5.0`** — you ran
  `pip install -e .` before installing the engine. `biofilm-processing` isn't on PyPI; run
  `bash scripts/setup.sh` (or `pip install -e external/biofilm-processing` first).
- **`external/biofilm-processing does not appear to be a Python project`** — the submodule
  is empty because the repo was cloned/pulled without submodules. Fix:
  `git submodule update --init --recursive`, then re-run the install.
- **Cloned without `--recurse-submodules` (e.g. via `git pull` on an existing checkout)?**
  Just run `git submodule update --init --recursive` once. Do **not** `git clone` the
  processing repo into the working dir by hand — it belongs only at `external/biofilm-processing`.

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

The sibling `~/embeddings/` repo runs in a conda env called `embeddings` with torch 2.4. That's too old for sm_120, and `pip install -e .` for this repo will try to upgrade it. If you want to keep the old env pinned and avoid disrupting `~/embeddings/`'s scripts, create a fresh env for biofilm-embeddings instead.

## Run

```bash
biofilm-embeddings-gui
```

### Desktop shortcut (optional)

`scripts/installDesktopShortcut.py` installs a launcher in the OS's application menu (and on the Desktop if one exists) using `assets/dora5.jpg` as the icon. The launcher auto-activates the conda env you ran the installer from.

```bash
python scripts/installDesktopShortcut.py
```

Works on Linux (`.desktop`), macOS (`.app` bundle), and Windows (`.bat` + `.lnk`). On macOS and Windows, JPG is not a valid icon format for Finder / Explorer respectively, so for those platforms drop a converted icon next to `dora5.jpg`:

- macOS: `sips -s format icns assets/dora5.jpg --out assets/dora5.icns`
- Windows: `magick convert assets/dora5.jpg -define icon:auto-resize=256,128,64,48,32,16 assets/dora5.ico`

Re-run the installer after converting. To remove later, delete `biofilm-embeddings.desktop` from `~/.local/share/applications/` (Linux), `biofilm-embeddings.app` from `~/Desktop/` (macOS), or `biofilm-embeddings.lnk` from the Desktop (Windows).

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

This repo layers the DINOv2 + dataset code (originally prototyped in `~/embeddings/`) on top
of `~/biofilm-processing`, which it imports directly via a pinned git submodule
(`external/biofilm-processing` @ v0.5.0) rather than copying. Processing is therefore a single
source of truth: `biofilm_embeddings` ships only the GUI + embeddings layer and calls
`multiWellAnalysis.processing` from the submodule, so the `_processed.tif` render can never
diverge from upstream. Colony tracking, whole-image, and intensity feature extraction live in
`biofilm-processing` and are out of scope here.
