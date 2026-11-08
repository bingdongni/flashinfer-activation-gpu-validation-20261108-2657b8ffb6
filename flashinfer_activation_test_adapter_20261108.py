"""Load FlashInfer's actual Triton sources without unrelated JIT dependencies."""

import os
import sys
import types
from pathlib import Path


def pytest_configure(config):
    root = Path(os.environ['FLASHINFER_SOURCE_ROOT']) / 'flashinfer'
    for name, path in [
        ('flashinfer', root),
        ('flashinfer.triton', root / 'triton'),
        ('flashinfer.triton.kernels', root / 'triton' / 'kernels'),
    ]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
