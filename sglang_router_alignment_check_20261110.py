import sglang_router_test_adapter_20261110  # noqa: F401
import torch
from sglang.kernels.ops.moe import moe_route_warp


class Arch:
    def __init__(self, major):
        self.major = major

    def is_sm_at_least(self, major):
        return self.major >= major


count = 0
for major in (8, 10):
    moe_route_warp.current_arch = lambda: Arch(major)
    for experts in range(128, 513, 32):
        for dtype in (torch.float32, torch.bfloat16):
            for rows in (0, 1, 3):
                for padding in (0, 1, 2, 4, 8):
                    for offset in (0, 1, 4, 8):
                        x = torch.empty((rows, experts + padding + offset), dtype=dtype, device="cuda")[:, offset : offset + experts]
                        lanes = experts // 32
                        vector = 1
                        cap = 32 if major >= 10 else 16
                        while lanes % (2 * vector) == 0 and 2 * vector * x.element_size() <= cap:
                            vector *= 2
                        alignment = vector * x.element_size()
                        native_accepts = (
                            x.stride(1) == 1 and x.data_ptr() % alignment == 0
                            and (rows == 1 or x.stride(0) * x.element_size() % alignment == 0)
                        )
                        assert moe_route_warp.scores_layout_ok(x) == native_accepts, (major, experts, dtype, rows, padding, offset)
                        count += 1
print(f"{count} GPU tensor layouts matched native matcher rules for simulated SM8 and SM10")
