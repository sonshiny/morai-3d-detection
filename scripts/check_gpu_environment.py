#!/usr/bin/env python3
"""Fail fast on a CUDA/PyTorch build that cannot execute on the installed GPU."""

import json
import re
import sys

import torch


def version_tuple(value):
    nums = re.findall(r"\d+", str(value))
    return tuple(int(x) for x in nums[:3])


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: driver/PyTorch installation을 먼저 수정하세요.")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    torch_v = version_tuple(torch.__version__)
    cuda_v = version_tuple(torch.version.cuda or "0")

    # Blackwell support entered stable PyTorch with 2.7 + CUDA 12.8 wheels.
    # Use runtime capability rather than trusting a marketing product name.
    if capability >= (12, 0) and (torch_v < (2, 7) or cuda_v < (12, 8)):
        raise SystemExit(
            "Blackwell-class GPU에는 PyTorch >=2.7 및 CUDA runtime >=12.8 build가 "
            f"필요합니다: torch={torch.__version__}, cuda={torch.version.cuda}, sm={capability}"
        )

    device = torch.device("cuda", index)
    model = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    x = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
    loss = model(x).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    record = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(index),
        "compute_capability": list(capability),
        "conv_forward_backward_finite": bool(torch.isfinite(loss).item()),
    }
    if not record["conv_forward_backward_finite"]:
        raise SystemExit("CUDA conv forward/backward produced non-finite result")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
