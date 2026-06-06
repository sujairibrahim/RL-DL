"""
mare/utils/exceptions.py
-------------------------
Custom exceptions for the MARE / REMARL framework.
"""


class MAREBaseError(Exception):
    """Base exception for all MARE errors."""
    pass


class ConfigurationError(MAREBaseError):
    """Raised when agent or system configuration is invalid."""

    def __init__(self, message: str, config_file: str = ""):
        super().__init__(message)
        self.config_file = config_file

    def __str__(self):
        base = super().__str__()
        if self.config_file:
            return f"{base}  [config: {self.config_file}]"
        return base


class AgentExecutionError(MAREBaseError):
    """Raised when an agent fails to execute an action."""

    def __init__(self, message: str, agent_name: str = "", action: str = ""):
        super().__init__(message)
        self.agent_name = agent_name
        self.action = action

    def __str__(self):
        base = super().__str__()
        parts = []
        if self.agent_name:
            parts.append(f"agent={self.agent_name}")
        if self.action:
            parts.append(f"action={self.action}")
        return f"{base}  [{', '.join(parts)}]" if parts else base


class WorkspaceError(MAREBaseError):
    """Raised when a workspace operation fails."""
    pass


class LLMError(MAREBaseError):
    """Raised when an LLM call fails."""
    pass
