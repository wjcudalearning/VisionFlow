"""Full-resolution traditional-CV detector tuning tool."""

from .engine import ContourProcessingEngine, ProcessingRecipe, ProcessingResult
from .version import __version__

__all__ = [
    "ContourProcessingEngine",
    "ProcessingRecipe",
    "ProcessingResult",
    "__version__",
]
