"""
mare/utils/logging.py
----------------------
Shared logging utilities for MARE agents.
"""

import logging
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Return a standard Python logger for the given module name."""
    return logging.getLogger(name)


class MARELoggerMixin:
    """
    Mixin that gives any class log_info / log_debug / log_warning / log_error
    convenience methods backed by a standard Python logger named after the class.
    """

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(
            f"mare.{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def log_info(self, msg: str) -> None:
        self._logger.info(msg)

    def log_debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def log_warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def log_error(self, msg: str) -> None:
        self._logger.error(msg)
