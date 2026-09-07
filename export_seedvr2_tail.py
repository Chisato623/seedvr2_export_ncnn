"""Export the SeedVR2 DiT output norm, modulation, and patch projection."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path

import torch
from torch import nn


DIM = 2560
EPSILON = 1e-5


def parameter(value: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(value, requires_grad=False)


class SeedVR2Tail(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.norm_weight = parameter(state["vid_out_norm.weight"])
        self.shift = parameter(state["vid_out_ada.out_shift"])
        self.scale = parameter(state["vid_out_ada.out_scale"])
        projection_weight = state["vid_out.proj.weight"]
        projection_bias = state["vid_out.proj.bias"]
        self.projection = nn.Linear(
            projection_weight.shape[1], projection_weight.shape[0], device="meta"
        )
        self.projection.weight = parameter(projection_weight)
        self.projection.bias = parameter(projection_bias)

    def forward(
        self,
        video: torch.Tensor,
        attention_shift: torch.Tensor,
        attention_scale: torch.Tensor,
    ) -> torch.Tensor:
        variance = video.float().square().mean(-1, keepdim=True)
        video = video * torch.rsqrt(variance + EPSILON).to(video.dtype)
        video = video * self.norm_weight
        # Official inference reuses Cache["emb_repeat_0_vid"] here. That cache
        # was populated by the first block's attention AdaSingle call, so the
        # dynamic output modulation is the attention shift/scale triplet.
        video = video * (attention_scale + self.scale)
        video = video + attention_shift + self.shift
        return self.projection(video)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true",
                        help="store NCNN weights in FP16 (default: FP32)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model = SeedVR2Tail(state).eval()
    example = (
        torch.zeros(8, DIM),
        torch.zeros(1, DIM),
        torch.zeros(1, DIM),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    traced_path = args.output_dir / "seedvr2_tail.pt"
    traced.save(traced_path)
    subprocess.run(
        [
            str(args.pnnx),
            str(traced_path),
            "inputshape=[8,2560],[1,2560],[1,2560]",
            "inputshape2=[12,2560],[1,2560],[1,2560]",
            f"fp16={int(args.fp16)}",
        ],
        cwd=args.output_dir,
        check=True,
    )

    manifest_path = args.output_dir.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tail"] = {
        "param": str(
            (args.output_dir / "seedvr2_tail.ncnn.param").relative_to(
                args.output_dir.parent
            )
        ),
        "bin": str(
            (args.output_dir / "seedvr2_tail.ncnn.bin").relative_to(
                args.output_dir.parent
            )
        ),
        "input_order": ["video", "attention_shift", "attention_scale"],
        "output": "patchified_prediction",
        "output_channels_per_patch": 64,
        "compatibility": (
            "Uses the first attention AdaSingle shift/scale triplet, matching "
            "the official emb_repeat_0_vid cache reuse at vid_out_ada."
        ),
    }
    manifest.pop("known_issues", None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    del traced, model
    gc.collect()
    for suffix in (".pt", ".pnnx.param", ".pnnx.bin", ".pnnx.onnx"):
        (args.output_dir / f"seedvr2_tail{suffix}").unlink(missing_ok=True)
    for suffix in ("_pnnx.py", "_ncnn.py"):
        (args.output_dir / f"seedvr2_tail{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
