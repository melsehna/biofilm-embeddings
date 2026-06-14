"""Smoke tests for the single-source integration with biofilm-processing.

biofilm-embeddings imports biofilm-processing's `multiWellAnalysis.processing`
(it no longer ships its own copy). These tests catch the two ways that contract
can break — loudly, on the first run / in CI, instead of silently corrupting
embeddings. See INTEGRATION_PLAN.md.
"""
import inspect

import pytest


def test_processing_is_biofilm_not_a_local_fork():
    """`multiWellAnalysis.processing` must resolve to biofilm-processing, not a
    re-introduced fork under biofilm_embeddings."""
    import multiWellAnalysis.processing.analysis_main as am
    path = am.__file__.replace('\\', '/')
    assert 'biofilm_embeddings' not in path, (
        f'processing resolved to {am.__file__} — a local fork crept back into '
        'the biofilm_embeddings package; it must import biofilm-processing, not copy it.'
    )
    assert 'biofilm-processing' in path, (
        f'processing resolved to {am.__file__} — expected it to come from the '
        'external/biofilm-processing submodule.'
    )


def test_timelapse_signature_contract():
    """Every keyword biofilm_embeddings's GUI passes to timelapseProcessing must
    still exist in biofilm-processing's signature (catches API drift)."""
    from multiWellAnalysis.processing.analysis_main import timelapseProcessing
    have = set(inspect.signature(timelapseProcessing).parameters)
    passed = {  # mirror of gui/tabs/run.py:_processOneWell's call
        'images', 'blockDiameter', 'ntimepoints', 'shiftThresh', 'fixedThresh',
        'dustCorrection', 'outdir', 'filename', 'imageRecords', 'fftStride',
        'downsample', 'skipOverlay', 'workers',
    }
    missing = passed - have
    assert not missing, (
        f'timelapseProcessing no longer accepts {missing}; update the call in '
        'biofilm_embeddings/gui/tabs/run.py to match biofilm-processing.'
    )


def test_metadata_and_preprocessing_entrypoints_exist():
    from multiWellAnalysis.processing.image_metadata import probePlateMeta  # noqa
    from multiWellAnalysis.processing.preprocessing import normalizeLocalContrast  # noqa


def test_embeddings_import():
    """biofilm_embeddings's own layer imports (needs torch — skipped if absent)."""
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    import biofilm_embeddings.embeddings.extractor  # noqa
    import biofilm_embeddings.embeddings.extract_one_plate  # noqa
