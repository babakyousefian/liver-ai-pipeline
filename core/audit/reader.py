# core/audit/reader.py
# -*- coding: utf-8 -*-
"""
Reader for audit log entries with filtering and pagination.

Usage:
    from core.audit.reader import read_filtered

    entries = read_filtered(
        level="WARNING",
        category="AUTH",
        user="babak",
        search="login",
        limit=100,
    )
"""

from datetime import datetime
from typing import List, Optional

from .models import AuditEntry


def read_filtered(
        entries: List[AuditEntry],
        level: Optional[str] = None,
        category: Optional[str] = None,
        user: Optional[str] = None,
        search: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: Optional[int] = None,
) -> List[AuditEntry]:
    """
    Filter entries by various criteria.

    Args:
        entries: list of AuditEntry
        level: filter by level (INFO, WARNING, ...)
        category: filter by category (AUTH, DATA, ...)
        user: filter by username
        search: search in action or details
        date_from: ISO date string (inclusive)
        date_to: ISO date string (inclusive)
        limit: max entries to return (most recent first)

    Returns:
        Filtered list (most recent first)
    """
    result = []
    search_lower = (search or "").lower().strip()
    date_from_dt = _parse_date(date_from)
    date_to_dt = _parse_date(date_to)

    for e in entries:
        # ✅ Level filter
        if level and e.level != level:
            continue
        # ✅ Category filter
        if category and e.category != category:
            continue
        # ✅ User filter
        if user and e.user != user:
            continue
        # ✅ Search filter
        if search_lower:
            haystack = (e.action or "").lower()
            try:
                import json
                haystack += " " + json.dumps(e.details, ensure_ascii=False, default=str).lower()
            except Exception:
                pass
            if search_lower not in haystack:
                continue
        # ✅ Date range
        if date_from_dt or date_to_dt:
            try:
                ts_dt = datetime.fromisoformat(e.timestamp)
            except Exception:
                continue
            if date_from_dt and ts_dt < date_from_dt:
                continue
            if date_to_dt and ts_dt > date_to_dt:
                continue
        result.append(e)

    # ✅ Sort most recent first
    result.sort(key=lambda x: x.timestamp, reverse=True)

    if limit is not None and limit > 0:
        result = result[:limit]

    return result


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        # try ISO date
        if len(date_str) == 10:
            return datetime.strptime(date_str, "%Y-%m-%d")
        return datetime.fromisoformat(date_str)
    except Exception:
        return None
