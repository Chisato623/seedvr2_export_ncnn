#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
value = torch.load(args.input, map_location="cpu", weights_only=True)
if not isinstance(value, torch.Tensor):
    raise TypeError(f"expected Tensor, got {type(value)!r}")
value = value.float().squeeze(0).contiguous()
if value.ndim != 2 or value.shape[1] != 5120:
    raise ValueError(f"expected [tokens,5120], got {tuple(value.shape)}")
value.numpy().tofile(args.output)
print(f"shape={tuple(value.shape)} output={args.output}")
