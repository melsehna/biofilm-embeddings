#!/usr/bin/env bash
#
# Embed-only launcher: generate DINOv2 embeddings from an ALREADY-PROCESSED
# output tree (skips image processing). Detached via nohup with a timestamped
# log, so it survives an SSH disconnect. Run on a GPU machine.
#
# Usage:
#   conda activate embeddings           # the env with biofilm-embeddings installed
#   bash scripts/extract_embeddings.sh /path/to/output_root [extra extract args...]
#
# Example (training set on the NAS — the mount path is per-machine):
#   # on the processing box, the share is mounted at /mnt/phenotyper:
#   bash scripts/extract_embeddings.sh \
#     /mnt/phenotyper/Sehna/multiphenotype-data-061426/trainingData
#   # on the GPU box, the same share is mounted at /mnt/bridgeslab/phenotyper:
#   bash scripts/extract_embeddings.sh \
#     /mnt/bridgeslab/phenotyper/Sehna/multiphenotype-data-061426/trainingData
#
# Writes <output_root>/embeddings/cls_cache.pt (resumable per-batch). Any extra
# args after the root are passed straight through to biofilm-embeddings-run
# (e.g. --model facebook/dinov2-giant, --well-batch 8, --dry-run).
#
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${1:-}"
if [[ -z "$ROOT" ]]; then
    echo "ERROR: pass the processed output root as the first argument." >&2
    echo "  e.g. bash scripts/extract_embeddings.sh /mnt/.../trainingData" >&2
    exit 2
fi
if [[ ! -d "$ROOT" ]]; then
    echo "ERROR: not a directory: $ROOT" >&2
    exit 2
fi
shift  # remaining args pass through to the extractor

TS=$(date +%Y%m%d_%H%M%S)
LOG="$(pwd)/scripts/extract_embeddings_${TS}.log"

# Prefer the installed console entry point; fall back to module form (PYTHONPATH=src)
# so it also works from a checkout without `pip install -e .`.
if command -v biofilm-embeddings-run >/dev/null 2>&1; then
    RUNNER=(biofilm-embeddings-run)
else
    RUNNER=(env PYTHONPATH=src python -m biofilm_embeddings.embeddings.extract_run)
fi

nohup "${RUNNER[@]}" "$ROOT" "$@" < /dev/null > "$LOG" 2>&1 &
PID=$!
echo
echo "Embedding run launched detached (survives logout)."
echo "  Root:     $ROOT"
echo "  PID:      $PID"
echo "  Log:      $LOG"
echo "  Monitor:  tail -f \"$LOG\""
echo "  Stop:     kill $PID"
echo "  Output:   $ROOT/embeddings/cls_cache.pt"
