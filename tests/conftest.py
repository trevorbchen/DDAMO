"""
Session-level setup that runs before any test module is collected.

torch._dynamo initializes lazily and probes sys.modules for packages like
pandas via importlib.util.find_spec.  If test_comparison_utils.py has already
replaced sys.modules["pandas"] with a MagicMock before this probe happens,
importlib raises ValueError.  Importing torch.optim here (which triggers the
_dynamo scan) ensures the probe completes before any module-level mocking.
"""
import torch
import torch.optim  # noqa: F401 — side-effect: initializes torch._dynamo
