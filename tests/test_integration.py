"""Smoke tests for the single-source integration with biofilm-processing.

microtyper-vision imports biofilm-processing's `multiWellAnalysis.processing`
(it no longer ships its own copy). These tests catch the two ways that contract
can break — loudly, on the first run / in CI, instead of silently corrupting
embeddings. See INTEGRATION_PLAN.md.
"""
import inspect

import pytest


def test_processing_is_biofilm_not_a_local_fork():
    """`multiWellAnalysis.processing` must resolve to biofilm-processing, not a
    re-introduced fork under microtyper_vision."""
    import multiWellAnalysis.processing.analysis_main as am
    assert 'microtyper' not in am.__file__.lower(), (
        f'processing resolved to {am.__file__} — a local fork crept back in; '
        'microtyper-vision must import biofilm-processing, not copy it.'
    )


def test_timelapse_signature_contract():
    """Every keyword microtyper_vision's GUI passes to timelapseProcessing must
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
        'microtyper_vision/gui/tabs/run.py to match biofilm-processing.'
    )


def test_metadata_and_preprocessing_entrypoints_exist():
    from multiWellAnalysis.processing.image_metadata import probePlateMeta  # noqa
    from multiWellAnalysis.processing.preprocessing import normalizeLocalContrast  # noqa


def test_embeddings_import():
    """microtyper_vision's own layer imports (needs torch — skipped if absent)."""
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    import microtyper_vision.embeddings.extractor  # noqa
    import microtyper_vision.embeddings.extract_one_plate  # noqa
