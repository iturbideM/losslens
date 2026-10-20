"""losslens: visualize the loss landscape around a trained PyTorch model."""
from .compare import compare
from .surface import Surface, load, surface

__all__ = ["surface", "compare", "load", "Surface"]
__version__ = "0.1.0"
