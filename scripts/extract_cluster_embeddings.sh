#!/usr/bin/env bash
#
# Reproducible record: DINOv2 embedding extraction for the CLUSTER set
# (Jesse "Project - Cluster": compound treatments + clean deletions, 10x,
# processed by biofilm-processing run_cluster_062726.sh).
#
# EMBED-ONLY: runs on the already-processed output tree — skips image processing.
# Walks every <plate>/processedImages/index.csv, embeds each well's _processed.tif,
# and writes <root>/embeddings/cls_cache.pt. Detached via nohup with a timestamped
# log, so it survives an SSH disconnect. Run on a GPU machine.
#
# DATA: 8 plates, 10x (_02), ~688 wells total (96/80/64 per plate). 41 FRAMES per
# well (uniform). Both this set and the reimaging set image at 1 frame/hour
# (verified from TIFF KineticStartTime), so this set's first 31 frames = the same
# 0-30 h window as the reimaging set's 31 frames.
#
# FRAME COUNT: defaults to --n-frames 31 so the embeddings are directly comparable
# to the reimaging/training sets (the extractor reads only the first 31 frames;
# the extra 10 = hours 31-40 are dropped). 0 wells skipped (all have >=31). To
# instead embed the full 41-frame timecourse (cluster-only analysis), override
# with --n-frames 41 on the command line (last value wins).
#
# SOURCE TREE: the cluster output on the NAS. The share mounts at different paths
# per machine, so the root is auto-detected (writable one wins):
#   processing box : /mnt/phenotyper/Sehna/...
#   GPU box        : /mnt/bridgeslab/phenotyper/Sehna/...
#
# COMPARABILITY: --model / --image-size / --grid-size MUST match the training and
# reimaging extractions (defaults dinov2-base / 518 / grid 3 — identical here).
# --well-batch is a perf-only knob (does NOT change embeddings); pick per GPU.
# With the default --n-frames 31 (below), the frame count AND the real-time window
# match the reimaging set, so a frames-as-features wide table is directly
# comparable across the two sets.
#
# LABELS: the embeddings carry the timestamp plate name (e.g. 260414_113437_Plate 1),
# not the experiment name. Join <root>/mapping.csv (plateID -> experiment) — or the
# master CSVs' drawerID column — to recover CmpdTreatment / cleanDeletion labels.
# Per-condition layouts (gene/compound) live in
# ~/biofilm-analysis/data/reimaging_updated/{compounds,cleanDeletions_hand,...}.
#
# Usage (defaults to --n-frames 31 for reimaging comparability):
#   conda activate <env>             # the env with biofilm-embeddings installed
#   bash scripts/extract_cluster_embeddings.sh --well-batch 24 --dry-run   # verify, no GPU
#   bash scripts/extract_cluster_embeddings.sh --well-batch 24             # full run (first 31 frames)
#   bash scripts/extract_cluster_embeddings.sh --well-batch 24 --n-frames 41   # full 41-frame timecourse
# For an arbitrary root, call the CLI directly: biofilm-embeddings-run /path/to/root
#
set -euo pipefail
cd "$(dirname "$0")/.."

# --- cluster output root: first existing+writable candidate wins (per-machine mount) ---
CANDIDATES=(
    "/mnt/bridgeslab/phenotyper/Sehna/cluster-data-062726/clusterData"
    "/mnt/phenotyper/Sehna/cluster-data-062726/clusterData"
)
ROOT=""
for c in "${CANDIDATES[@]}"; do
    [[ -d "$c" && -w "$c" ]] && { ROOT="$c"; break; }
done
if [[ -z "$ROOT" ]]; then
    echo "ERROR: no writable cluster output root at known mounts:" >&2
    printf '  %s\n' "${CANDIDATES[@]}" >&2
    echo "Is the share mounted read-write here? For a custom path / read-only source" >&2
    echo "with a separate cache dir: biofilm-embeddings-run /path/to/root --cache-dir /local/dir" >&2
    exit 2
fi
echo "Cluster output root: $ROOT"

# --- launch detached, timestamped log named for this dataset ---
TS=$(date +%Y%m%d_%H%M%S)
LOG="$(pwd)/scripts/extract_cluster_embeddings_${TS}.log"

if command -v biofilm-embeddings-run >/dev/null 2>&1; then
    RUNNER=(biofilm-embeddings-run)
else
    RUNNER=(env PYTHONPATH=src python -m biofilm_embeddings.embeddings.extract_run)
fi

# --n-frames 31 first so the run is reimaging-comparable by default; "$@" after it
# means a user-supplied --n-frames (e.g. 41) overrides (argparse: last value wins).
nohup "${RUNNER[@]}" "$ROOT" --n-frames 31 "$@" < /dev/null > "$LOG" 2>&1 &
PID=$!
echo
echo "Cluster embedding run launched detached (survives logout)."
echo "  Root:     $ROOT"
echo "  PID:      $PID"
echo "  Log:      $LOG"
echo "  Monitor:  tail -f \"$LOG\""
echo "  Stop:     kill $PID"
echo "  Output:   $ROOT/embeddings/cls_cache.pt"
