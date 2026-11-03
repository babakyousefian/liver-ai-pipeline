# core/updater/models.py
# -*- coding: utf-8 -*-
"""
Data models for the updater.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime


@dataclass
class UpdateInfo:
    """Information about an available update."""
    version: str                          # e.g. "3.1.0"
    release_date: Optional[str] = None    # ISO string
    changelog: str = ""                   # Markdown
    download_url: str = ""                # Direct URL to asset
    download_size: int = 0                # Bytes
    checksum_sha256: str = ""             # hex string
    is_prerelease: bool = False
    is_critical: bool = False             # Forced update flag
    min_os_version: str = ""              # Optional OS requirement
    asset_name: str = ""                  # e.g. "LiverAI.exe"

    @property
    def human_size(self) -> str:
        """e.g. '12.3 MB'"""
        size = float(self.download_size)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024.0 or unit == "GB":
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} GB"


@dataclass
class UpdateConfig:
    """User-configurable update settings."""
    auto_check: bool = True
    check_interval_hours: int = 24
    last_check: Optional[str] = None       # ISO timestamp
    skipped_version: str = ""              # User clicked "skip"
    source: str = "github"                # "github" | "website"
    github_repo: str = "babakyousefian/liver-ai-pipeline"
    website_url: str = "https://babakyousefian.com/api/latest.json"
    include_prerelease: bool = False
    channel: str = "stable"               # "stable" | "beta" | "alpha"

    def to_dict(self) -> Dict:
        return {
            "auto_check": self.auto_check,
            "check_interval_hours": self.check_interval_hours,
            "last_check": self.last_check,
            "skipped_version": self.skipped_version,
            "source": self.source,
            "github_repo": self.github_repo,
            "website_url": self.website_url,
            "include_prerelease": self.include_prerelease,
            "channel": self.channel,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UpdateConfig":
        obj = cls()
        for key, value in (data or {}).items():
            if hasattr(obj, key) and value is not None:
                setattr(obj, key, value)
        return obj