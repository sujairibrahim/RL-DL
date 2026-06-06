"""
mare/__init__.py
Public API for the MARE package.

Note: agent imports are intentionally deferred to avoid circular imports
during package initialisation. Import agents directly from their modules
or via mare.agents.
"""

from mare.workspace.shared_workspace import SharedWorkspace
from mare.utils.llm_client import LLMClient

__all__ = ["SharedWorkspace", "LLMClient"]
