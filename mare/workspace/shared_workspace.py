"""
remarl/mare/workspace/shared_workspace.py
------------------------------------------
Thin wrapper around MARE's original SharedWorkspace.
If MARE's actual SharedWorkspace is importable, this delegates to it.
If not (e.g. during testing), it falls back to a dict-based mock.

Your real MARE SharedWorkspace from the repo should sit at:
    mare/workspace/shared_workspace.py  (i.e. this file IS the real one,
    or it delegates to MARE's internal implementation)

The contract REMARL depends on:
    workspace.get(key: str, default: str = "") -> str
    workspace.set(key: str, value: str)
    workspace.current_phase: int   (0=elicitation ... 4=done)
    workspace.get_srs() -> str     (returns full SRS text)
"""

import logging

logger = logging.getLogger(__name__)


class SharedWorkspace:
    """
    REMARL-compatible SharedWorkspace.

    This is the reference implementation used when MARE's own
    SharedWorkspace is not available.  Replace the body of this class
    with an import from MARE once you have cloned the MARE repo:

        from mare.workspace.shared_workspace import SharedWorkspace  # noqa
    """

    # Keys recognised as "version-controlled" artifacts
    ARTIFACT_KEYS = [
        "rough_idea",
        "initial_context",
        "user_stories",
        "questions",
        "req_draft",
        "req_model",
        "error_report",
        "srs_document",
    ]

    def __init__(self):
        self._store: dict = {}
        self.current_phase: int = 0   # 0-4

    # ── Core API ──────────────────────────────────────────────────────

    def get(self, key: str, default: str = "") -> str:
        return self._store.get(key, default)

    def set(self, key: str, value: str):
        self._store[key] = value

    def append(self, key: str, value: str, separator: str = "\n"):
        """Append text to an existing field (convenience helper)."""
        existing = self._store.get(key, "")
        self._store[key] = (existing + separator + value).strip()

    def update(self, updates: dict):
        """Batch update multiple fields."""
        self._store.update(updates)

    # ── SRS export ────────────────────────────────────────────────────

    def get_srs(self) -> str:
        """Return the final SRS text (used by Oracle)."""
        srs = self.get("srs_document")
        if srs:
            return srs
        # Fallback: concatenate all artifact fields
        parts = [self.get(k) for k in self.ARTIFACT_KEYS if self.get(k)]
        return "\n\n".join(parts)

    # ── Phase management ─────────────────────────────────────────────

    def advance_phase(self):
        self.current_phase = min(self.current_phase + 1, 4)

    def is_complete(self) -> bool:
        return self.current_phase == 4 and bool(self.get("srs_document"))

    # ── Debug helpers ─────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [f"SharedWorkspace (phase={self.current_phase})"]
        for k in self.ARTIFACT_KEYS:
            v = self.get(k)
            if v:
                lines.append(f"  {k:<20} {len(v):>5} chars")
        return "\n".join(lines)

    def __repr__(self):
        filled = [k for k in self.ARTIFACT_KEYS if self.get(k)]
        return f"SharedWorkspace(phase={self.current_phase}, fields={filled})"
