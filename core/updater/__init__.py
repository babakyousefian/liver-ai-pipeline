# core/updater/__init__.py
# -*- coding: utf-8 -*-

from .version import (
    parse_version, is_newer, is_same_major, format_version,
)
from .models import UpdateInfo, UpdateConfig
from .source import make_source, GitHubSource, WebsiteSource
from .checker import UpdateChecker, CURRENT_VERSION
from .downloader import UpdateDownloader
from .verifier import verify_checksum, verify_size, sha256_of_file
from .installer import install_frozen, install_source, is_frozen, get_app_dir

__all__ = [
    "parse_version", "is_newer", "is_same_major", "format_version",
    "UpdateInfo", "UpdateConfig",
    "make_source", "GitHubSource", "WebsiteSource",
    "UpdateChecker", "CURRENT_VERSION",
    "UpdateDownloader",
    "verify_checksum", "verify_size", "sha256_of_file",
    "install_frozen", "install_source", "is_frozen", "get_app_dir",
]