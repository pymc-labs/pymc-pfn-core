"""pymc-pfn: amortized posterior networks from a simulator or a PyMC model."""

from pymc_pfn.audit import AuditReport, PriorAuditError
from pymc_pfn.prior import Prior
from pymc_pfn.source import Simulator, TaskSource
from pymc_pfn.task import TaskBatch, concat

__all__ = [
    "PFN",
    "AuditReport",
    "FitConfig",
    "ModelConfig",
    "Posterior",
    "Prior",
    "PriorAuditError",
    "Report",
    "Simulator",
    "Symmetry",
    "TaskBatch",
    "TaskSource",
    "concat",
    "evaluate",
    "fit",
]

_LAZY = {
    "evaluate": "pymc_pfn.evaluation",
    "Report": "pymc_pfn.evaluation",
    "Symmetry": "pymc_pfn.evaluation",
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
