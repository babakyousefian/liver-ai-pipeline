# core/updater/version.py
# -*- coding: utf-8 -*-
"""
Version parsing and comparison utilities.

Semantic Versioning: MAJOR.MINOR.PATCH[-prerelease][+build]
Examples:
    1.0.0
    3.0.10
    2.5.0-beta.1
    2.5.0+build.123
"""

import re
from typing import Tuple, Optional


_VERSION_RE = re.compile(
    r"^v?(?P<major>\d+)"          # v3 or 3
    r"(?:\.(?P<minor>\d+))?"      # .0
    r"(?:\.(?P<patch>\d+))?"      # .10
    r"(?:-(?P<pre>[0-9A-Za-z.\-]+))?"   # -beta.1
    r"(?:\+(?P<build>[0-9A-Za-z.\-]+))?$"  # +build.123
)


def parse_version(version: str) -> Tuple[int, int, int, str]:
    """
    Parse a version string into (major, minor, patch, prerelease).

    The prerelease part is normalized for sorting:
    - Empty string (release) sorts HIGHER than any prerelease.
    - "beta" < "rc" < "" (release)

    Raises:
        ValueError: if version string is invalid.
    """
    if not version or not isinstance(version, str):
        raise ValueError(f"Invalid version: {version!r}")

    m = _VERSION_RE.match(version.strip())
    if not m:
        raise ValueError(f"Invalid version format: {version!r}")

    major = int(m.group("major") or 0)
    minor = int(m.group("minor") or 0)
    patch = int(m.group("patch") or 0)

    pre = m.group("pre") or ""
    # For sorting: release ("") must be GREATER than prerelease.
    # So we add a high sentinel for release.
    # e.g., (1,0,0,"~") > (1,0,0,"beta.1")
    pre_key = pre if pre else "\uffff"  # highest char

    return (major, minor, patch, pre_key)


def is_newer(current: str, remote: str) -> bool:
    """
    Return True if `remote` is strictly newer than `current`.
    """
    try:
        return parse_version(remote) > parse_version(current)
    except ValueError:
        # If we can't parse, assume NOT newer (safer default)
        return False


def is_same_major(current: str, remote: str) -> bool:
    """True if both have same MAJOR version (for breaking-change decisions)."""
    try:
        return parse_version(current)[0] == parse_version(remote)[0]
    except ValueError:
        return False


def format_version(version: str) -> str:
    """Normalize display: 'v3.0.10' → '3.0.10'."""
    return version.lstrip("vV").strip()