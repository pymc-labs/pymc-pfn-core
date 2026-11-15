"""pymc-pfn: write the prior as a PyMC model, get an amortized posterior network."""

from pymc_pfn.audit import AuditReport, PriorAuditError
from pymc_pfn.prior import Prior
from pymc_pfn.task import TaskBatch, concat

__all__ = [
    "PFN",
    "AuditReport",
    "FitConfig",
    "ModelConfig",
    "Posterior",
    "Prior",
    "PriorAuditError",
    "TaskBatch",
    "concat",
    "fit",
]

_LAZY = {
    "fit": "pymc_pfn.training",
    "FitConfig": "pymc_pfn.training",
    "ModelConfig": "pymc_pfn.model",
    "PFN": "pymc_pfn.predict",
    "Posterior": "pymc_pfn.predict",
}


def __getattr__(name: str):
    # torch is imported only when the network is used.
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'pymc_pfn' has no attribute {name!r}")
