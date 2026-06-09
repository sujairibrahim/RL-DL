"""remarl/eval/__init__.py

FIX: Added compare_with_significance to __all__ — it was previously missing,
     so `from eval import compare_with_significance` would raise ImportError.
"""
from eval.metrics import (
    EvalResult,
    aggregate_oracle_results,
    print_comparison,
    compare_with_significance,   # FIX: was not exported
)

__all__ = [
    "EvalResult",
    "aggregate_oracle_results",
    "print_comparison",
    "compare_with_significance",
]