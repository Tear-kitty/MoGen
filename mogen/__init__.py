"""MoGen refactored package."""

__all__ = ["MoGenAdapterXL", "MoGenProjection"]


def __getattr__(name):
    if name == "MoGenAdapterXL":
        from .pipeline import MoGenAdapterXL
        return MoGenAdapterXL
    if name == "MoGenProjection":
        from .projection import MoGenProjection
        return MoGenProjection
    raise AttributeError(name)
