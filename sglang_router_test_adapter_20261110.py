"""Run selected real SGLang router code on Windows without unrelated server/JIT imports."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import functools
import os
import sys
import types
from pathlib import Path

import torch
import triton.language as tl

# The local GPU is SM89, where PDL is disabled. Triton 3.2 lacks these later names.
tl.extra.cuda.gdc_wait = lambda: None
tl.extra.cuda.gdc_launch_dependents = lambda: None

ROOT = Path(os.environ.get("SGLANG_ROUTER_SOURCE_ROOT", r"C:\PR\sglang-scout-20261110")) / "python" / "sglang"


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


for name, path in [
    ("sglang", ROOT),
    ("sglang.kernels", ROOT / "kernels"),
    ("sglang.kernels.jit", ROOT / "kernels" / "jit"),
    ("sglang.kernels.ops", ROOT / "kernels" / "ops"),
    ("sglang.kernels.ops.moe", ROOT / "kernels" / "ops" / "moe"),
    ("sglang.srt", ROOT / "srt"),
    ("sglang.srt.layers", ROOT / "srt" / "layers"),
    ("sglang.srt.layers.moe", ROOT / "srt" / "layers" / "moe"),
    ("sglang.test", ROOT / "test"),
    ("sglang.test.ci", ROOT / "test" / "ci"),
]:
    package(name, path)


class CudaArch:
    cuda = None

    def __init__(self):
        self.cuda = self

    def is_sm_at_least(self, major, minor=0):
        return (8, 9) >= (major, minor)

    def is_sm(self, major, minor=None):
        return major == 8 and (minor is None or minor == 9)


arch = types.ModuleType("sglang.kernels.arch")
arch.current_arch = lambda: CudaArch()
sys.modules[arch.__name__] = arch

utils = types.ModuleType("sglang.kernels.jit.utils")
utils.cache_once = functools.cache
utils.is_arch_support_pdl = lambda: False
utils.make_cpp_args = lambda *args: args
utils.load_jit = lambda *args, **kwargs: None
utils.parallel_compile = lambda variants: None
utils.get_ci_test_range = lambda full, small: full
sys.modules[utils.__name__] = utils


class Flag:
    def __init__(self):
        self.value = False

    def get(self):
        return self.value

    @contextmanager
    def override(self, value):
        old = self.value
        self.value = value
        try:
            yield
        finally:
            self.value = old


environ = types.ModuleType("sglang.srt.environ")
environ.envs = types.SimpleNamespace(
    SGLANG_OPT_SM120_MTP_ROUTER_TOP22=Flag(), SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=Flag()
)
sys.modules[environ.__name__] = environ

ci = types.ModuleType("sglang.test.ci.ci_register")
ci.register_cuda_ci = lambda **kwargs: None
sys.modules[ci.__name__] = ci

from sglang.kernels.ops.moe import moe_route_radix  # noqa: E402

moe_route_radix.covered = lambda *args: False

# Extract the two real topk functions without importing the rest of the server.
path = ROOT / "srt" / "layers" / "moe" / "topk.py"
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in {"biased_grouped_topk_gpu", "biased_grouped_topk_impl"}
]
assert len(functions) == 2
for node in functions:
    node.decorator_list = []  # use the real reference body without torch.compile
module = types.ModuleType("sglang.srt.layers.moe.topk")
module.__dict__.update({
    "torch": torch,
    "F": torch.nn.functional,
    "envs": environ.envs,
    "_is_cuda": True,
    "_is_hip": False,
    "_use_aiter": False,
    "_is_sail": False,
    "_RENORMALIZE_SUM_EPSILON": 1e-20,
    "is_power_of_two": lambda n: n > 0 and n & (n - 1) == 0,
    "fused_topk_deepseek": lambda *args, **kwargs: None,
})
prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
compiled = compile(ast.fix_missing_locations(ast.Module(body=[prefix, *functions], type_ignores=[])), str(path), "exec")
exec(compiled, module.__dict__)
sys.modules[module.__name__] = module
