"""Resolve a human-readable build identifier for display in the GUI.

Combines the installed package version with git branch / short commit
information when the package is running from a source checkout. If git
isn't available (e.g., installed via pip from a wheel), falls back to
the package version alone.
"""

import os
import subprocess
from pathlib import Path

from .. import __version__


def _gitOutput(args, cwd):
    try:
        result = subprocess.run(
            ['git'] + list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _packageRoot():
    return Path(__file__).resolve().parent.parent.parent.parent


def gitInfo():
    """Return (branch, shortCommit, dirty) or None if not in a git repo."""
    root = _packageRoot()
    if not (root / '.git').exists():
        return None
    branch = _gitOutput(['rev-parse', '--abbrev-ref', 'HEAD'], cwd=root)
    short = _gitOutput(['rev-parse', '--short', 'HEAD'], cwd=root)
    if branch is None or short is None:
        return None
    status = _gitOutput(['status', '--porcelain'], cwd=root)
    dirty = bool(status)
    return branch, short, dirty


def buildString():
    """One-line build identifier for the title bar / status line.

    Examples:
        "v0.3.0-dev  ·  feature/segmentation-methods @ a5f0ce9*"
        "v0.3.0-dev  ·  main @ 31b798a"
        "v0.3.0-dev"   (no git checkout)
    """
    parts = [f'v{__version__}']
    info = gitInfo()
    if info is not None:
        branch, short, dirty = info
        marker = '*' if dirty else ''
        parts.append(f'{branch} @ {short}{marker}')
    return '  ·  '.join(parts)
