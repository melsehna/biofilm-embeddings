#!/usr/bin/env bash
# One-shot setup for biofilm-embeddings.
#
# The image-processing engine (biofilm-processing) is NOT on PyPI — it is
# vendored as a pinned git submodule at external/biofilm-processing. It must be
# (1) fetched and (2) pip-installed editable BEFORE this package, otherwise
# `pip install -e .` cannot satisfy `biofilm-processing==0.5.0`. This script
# does both in the right order. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Fetching the pinned processing submodule (external/biofilm-processing)…"
# Sync first so a stale local URL (e.g. an old SSH remote) is replaced by the
# HTTPS URL pinned in .gitmodules before we try to fetch.
git submodule sync --recursive
git submodule update --init --recursive

if [ ! -f external/biofilm-processing/pyproject.toml ]; then
  echo "ERROR: external/biofilm-processing is still empty after submodule update."
  echo "       Check that .gitmodules points at the biofilm-processing repo and"
  echo "       that you have access to it, then re-run this script."
  exit 1
fi

echo "==> Installing the processing engine first (editable)…"
pip install -e external/biofilm-processing

echo "==> Installing biofilm-embeddings (editable)…"
pip install -e .

echo "==> Done. Launch the GUI with:  biofilm-embeddings-gui"
