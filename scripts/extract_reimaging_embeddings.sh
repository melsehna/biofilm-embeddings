#!/usr/bin/env bash
#
# Reproducible record: DINOv2 embedding extraction for the REIMAGING set
# (TN-Library reimaging, 10x, processed by biofilm-processing run_reimaging_062526.sh).
#
# EMBED-ONLY: runs on the already-processed output tree — skips image processing.
# Walks every <plate>/processedImages/index.csv, embeds each well's _processed.tif,
# and writes <root>/embeddings/cls_cache.pt. Detached via nohup with a timestamped
# log, so it survives an SSH disconnect. Run on a GPU machine.
#
# RUN ONLY AFTER PROCESSING COMPLETES: the reimaging run mirrors plates to the NAS
# as they finish; embedding a still-growing tree would miss the not-yet-mirrored
# plates. Wait for run_reimaging_062526.sh to finish (all 48 plates) before running.
#
# SOURCE TREE: the reimaging output on the NAS. The share mounts at different paths
# per machine, so the root is auto-detected:
#   processing box : /mnt/phenotyper/Sehna/...
#   GPU box        : /mnt/bridgeslab/phenotyper/Sehna/...
#
# COMPARABILITY: --model / --image-size / --grid-size MUST match the TRAINING
# extraction (extract_training_embeddings.sh) or the two embedding sets are not
# comparable. Defaults (dinov2-base, 518, grid 3) are identical in both scripts —
# only override them in BOTH or neither.
#
# Usage:
#   conda activate embeddings        # the env with biofilm-embeddings installed
#   bash scripts/extract_reimaging_embeddings.sh                 # full run
#   bash scripts/extract_reimaging_embeddings.sh --dry-run       # verify discovery, no GPU
#   bash scripts/extract_reimaging_embeddings.sh --well-batch 16 # extra flags pass through
# For an arbitrary root, call the CLI directly: biofilm-embeddings-run /path/to/root
#
set -euo pipefail
cd "$(dirname "$0")/.."

# --- reimaging output root: first existing candidate wins (per-machine mount) ---
CANDIDATES=(
    "/mnt/bridgeslab/phenotyper/Sehna/multiphenotype-data-06-25-26/reimagingData"
    "/mnt/phenotyper/Sehna/multiphenotype-data-06-25-26/reimagingData"
)
# Require the root be WRITABLE (the cache is written to <root>/embeddings/).
# This also disambiguates machines where the same share is mounted twice — e.g.
# the processing box has both /mnt/bridgeslab/phenotyper (read-only) and
# /mnt/phenotyper (writable); we want the writable one.
ROOT=""
for c in "${CANDIDATES[@]}"; do
    [[ -d "$c" && -w "$c" ]] && { ROOT="$c"; break; }
done
if [[ -z "$ROOT" ]]; then
    echo "ERROR: no writable reimaging output root at known mounts:" >&2
    printf '  %s\n' "${CANDIDATES[@]}" >&2
    echo "Has run_reimaging_062526.sh finished and mirrored to the NAS, and is the" >&2
    echo "share mounted read-write here? For a custom path / read-only source with a" >&2
    echo "separate cache dir: biofilm-embeddings-run /path/to/root --cache-dir /local/dir" >&2
    exit 2
fi
echo "Reimaging output root: $ROOT"

# --- launch detached, timestamped log named for this dataset ---
TS=$(date +%Y%m%d_%H%M%S)
LOG="$(pwd)/scripts/extract_reimaging_embeddings_${TS}.log"

# Installed console entry point if present; else module form from a checkout.
if command -v biofilm-embeddings-run >/dev/null 2>&1; then
    RUNNER=(biofilm-embeddings-run)
else
    RUNNER=(env PYTHONPATH=src python -m biofilm_embeddings.embeddings.extract_run)
fi

nohup "${RUNNER[@]}" "$ROOT" "$@" < /dev/null > "$LOG" 2>&1 &
PID=$!
echo
echo "Reimaging embedding run launched detached (survives logout)."
echo "  Root:     $ROOT"
echo "  PID:      $PID"
echo "  Log:      $LOG"
echo "  Monitor:  tail -f \"$LOG\""
echo "  Stop:     kill $PID"
echo "  Output:   $ROOT/embeddings/cls_cache.pt"
