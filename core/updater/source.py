# core/updater/source.py
# -*- coding: utf-8 -*-
"""
Update sources: GitHub Releases + custom Website endpoint.

All sources implement the same interface:
    fetch_latest(include_prerelease: bool) -> Optional[UpdateInfo]
"""

import json
import requests
from abc import ABC, abstractmethod
from typing import Optional

from .models import UpdateInfo


# ✅ Domain whitelist — prevent malicious redirects
_ALLOWED_DOMAINS = {
    "github.com",
    "api.github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "babakyousefian.com",
    "www.babakyousefian.com",
}

_TIMEOUT = 15  # seconds
_USER_AGENT = "LiverAI-Updater/1.0 (+https://babakyousefian.com)"


def _is_safe_url(url: str) -> bool:
    """Reject URLs that aren't HTTPS or aren't in the whitelist."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        if p.scheme not in ("https",):
            return False
        host = (p.hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in _ALLOWED_DOMAINS)
    except Exception:
        return False


class UpdateSource(ABC):
    """Abstract base for update sources."""

    @abstractmethod
    def fetch_latest(self, include_prerelease: bool = False) -> Optional[UpdateInfo]:
        """
        Fetch the latest release info.
        Returns None on network error / no release / invalid data.
        MUST NOT raise.
        """
        ...

    @staticmethod
    def _safe_get(url: str, headers: Optional[dict] = None, **kwargs) -> Optional[requests.Response]:
        """Safe GET with domain whitelist + timeout + error swallow."""
        if not _is_safe_url(url):
            print(f"⚠️ Updater: rejected unsafe URL: {url}")
            return None
        h = {"User-Agent": _USER_AGENT}
        if headers:
            h.update(headers)
        try:
            return requests.get(url, headers=h, timeout=_TIMEOUT, **kwargs)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Updater: network error: {type(e).__name__}: {e}")
            return None


class GitHubSource(UpdateSource):
    """GitHub Releases API source."""

    API_URL = "https://api.github.com/repos/{repo}/releases"

    def __init__(self, repo: str):
        self.repo = repo

    def fetch_latest(self, include_prerelease: bool = False) -> Optional[UpdateInfo]:
        url = self.API_URL.format(repo=self.repo)
        resp = self._safe_get(url)
        if resp is None or resp.status_code != 200:
            if resp is not None:
                if resp.status_code == 403:
                    print("⚠️ Updater: GitHub rate limit exceeded")
                elif resp.status_code == 404:
                    print(f"⚠️ Updater: repo not found: {self.repo}")
            return None

        try:
            releases = resp.json()
        except Exception:
            return None

        if not isinstance(releases, list) or not releases:
            return None

        # Pick first matching release (GitHub returns newest first)
        chosen = None
        for rel in releases:
            if rel.get("draft"):
                continue
            if rel.get("prerelease") and not include_prerelease:
                continue
            chosen = rel
            break

        if not chosen:
            return None

        return self._parse_release(chosen)

    def _parse_release(self, rel: dict) -> Optional[UpdateInfo]:
        try:
            tag = rel.get("tag_name", "")
            version = tag.lstrip("vV").strip()
            if not version:
                return None

            # Pick the primary asset (prefer .exe, then .zip, then .tar.gz)
            assets = rel.get("assets") or []
            asset = self._pick_asset(assets)
            if not asset:
                print(f"⚠️ Updater: no suitable asset in release {version}")
                return None

            download_url = asset.get("browser_download_url", "")
            if not _is_safe_url(download_url):
                print(f"⚠️ Updater: unsafe download URL: {download_url}")
                return None

            # Extract SHA256 from release body (format: `sha256: <hex>` or `SHA256: <hex>`)
            body = rel.get("body", "") or ""
            checksum = self._extract_checksum(body)

            return UpdateInfo(
                version=version,
                release_date=rel.get("published_at"),
                changelog=body,
                download_url=download_url,
                download_size=int(asset.get("size", 0)),
                checksum_sha256=checksum,
                is_prerelease=bool(rel.get("prerelease")),
                is_critical="[critical]" in body.lower() or "[security]" in body.lower(),
                asset_name=asset.get("name", ""),
            )
        except Exception as e:
            print(f"⚠️ Updater: parse release failed: {e}")
            return None

    @staticmethod
    def _pick_asset(assets: list) -> Optional[dict]:
        """Prefer .exe, then .zip, then .tar.gz."""
        priority = (".exe", ".zip", ".tar.gz", ".msi", ".dmg")
        for ext in priority:
            for a in assets:
                name = (a.get("name") or "").lower()
                if name.endswith(ext):
                    return a
        return assets[0] if assets else None

    @staticmethod
    def _extract_checksum(body: str) -> str:
        """Extract sha256 from release body: 'sha256: <hex>' or 'SHA256: <hex>'."""
        import re
        m = re.search(r"(?:sha256|SHA256|checksum)\s*[:=]\s*([a-fA-F0-9]{64})", body)
        return m.group(1).lower() if m else ""


class WebsiteSource(UpdateSource):
    """
    Custom website endpoint.

    Expected JSON response:
    {
        "version": "3.1.0",
        "release_date": "2026-11-03",
        "changelog": "## What's new...",
        "download_url": "https://babakyousefian.com/downloads/LiverAI-3.1.0.exe",
        "download_size": 12345678,
        "checksum_sha256": "abc123...",
        "is_prerelease": false,
        "is_critical": false,
        "asset_name": "LiverAI-3.1.0.exe"
    }
    """

    def __init__(self, url: str):
        self.url = url

    def fetch_latest(self, include_prerelease: bool = False) -> Optional[UpdateInfo]:
        resp = self._safe_get(self.url)
        if resp is None or resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        if data.get("is_prerelease") and not include_prerelease:
            return None

        download_url = data.get("download_url", "")
        if not _is_safe_url(download_url):
            print(f"⚠️ Updater: unsafe download URL: {download_url}")
            return None

        try:
            return UpdateInfo(
                version=str(data.get("version", "")).lstrip("vV"),
                release_date=data.get("release_date"),
                changelog=data.get("changelog", ""),
                download_url=download_url,
                download_size=int(data.get("download_size", 0)),
                checksum_sha256=(data.get("checksum_sha256") or "").lower(),
                is_prerelease=bool(data.get("is_prerelease")),
                is_critical=bool(data.get("is_critical")),
                asset_name=data.get("asset_name", ""),
            )
        except Exception as e:
            print(f"⚠️ Updater: parse website response failed: {e}")
            return None


def make_source(config) -> UpdateSource:
    """Factory: build the right source from config."""
    if config.source == "website":
        return WebsiteSource(config.website_url)
    return GitHubSource(config.github_repo)