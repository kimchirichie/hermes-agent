"""Fleet-wide kanban caps, read from the root home and never a profile's own."""
from __future__ import annotations

from typing import Any


def _root_kanban_section() -> dict:
    """``kanban:`` from the pre-profile (root) home's config, ``{}`` when unreadable."""
    try:
        from hermes_constants import get_default_hermes_root
        from hermes_cli.config import read_user_config_raw

        path = get_default_hermes_root() / "config.yaml"
        if not path.exists():
            return {}
        raw = read_user_config_raw(path)
        section = raw.get("kanban") if isinstance(raw, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:  # a policy read must never break card creation
        return {}


def fleet_caps() -> tuple[Any, Any]:
    """``(max_in_progress_per_profile, group_caps)`` from the root home, never a profile's own.

    The sparks are one shared resource, so their caps are one fleet number. A copy in a
    profile's ``config.yaml`` silently wins over the root and drifts (three profiles carried a
    group of 6 while the root said 3), so the dispatcher reads these two keys here only.
    """
    section = _root_kanban_section()
    return section.get("max_in_progress_per_profile"), section.get("group_caps") or None
