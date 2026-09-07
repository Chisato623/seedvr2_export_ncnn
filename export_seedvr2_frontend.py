"""Export SeedVR2 patch/text/timestep projections before the 32 DiT blocks."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DIM = 2560
EMBED_DIM = 15360


def parameter(value: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(value, requires_grad=False)


def linear_from_state(state: dict[str, torch.Tensor], prefix: str) -> nn.Linear:
    weight = state[f"{prefix}.weight"]
    bias = state[f"{prefix}.bias"]
    layer = nn.Linear(weight.shape[1], weight.shape[0], device="meta")
    layer.weight = parameter(weight)
    layer.bias = parameter(bias)
    return layer


class SeedVR2Frontend(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.video_projection = linear_from_state(state, "vid_in.proj")
        self.text_projection = linear_from_state(state, "txt_in")
        self.time_in = linear_from_state(state, "emb_in.proj_in")
        self.time_hidden = linear_from_state(state, "emb_in.proj_hid")
        self.time_out = linear_from_state(state, "emb_in.proj_out")

    def forward(
        self,
        patchified_video: torch.Tensor,
        text_embedding: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        video = self.video_projection(patchified_video)
        text = self.text_projection(text_embedding)
        embedding = self.time_out(
            F.silu(self.time_hidden(F.silu(self.time_in(timestep_embedding))))
        )
        modulation = embedding.reshape(-1, DIM, 2, 3)
        return (
            video,
            text,
            modulation[:, :, 0, 0],
            modulation[:, :, 0, 1],
            modulation[:, :, 0, 2],
            modulation[:, :, 1, 0],
            modulation[:, :, 1, 1],
            modulation[:, :, 1, 2],
        )


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
    model = SeedVR2Frontend(state).eval()
    example = (torch.zeros(8, 132), torch.zeros(4, 5120), torch.zeros(1, 256))
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    traced_path = args.output_dir / "seedvr2_frontend.pt"
    traced.save(traced_path)
    subprocess.run(
        [
            str(args.pnnx),
            str(traced_path),
            "inputshape=[8,132],[4,5120],[1,256]",
            "inputshape2=[12,132],[5,5120],[1,256]",
            f"fp16={int(args.fp16)}",
        ],
        cwd=args.output_dir,
        check=True,
    )
    manifest_path = args.output_dir.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frontend"] = {
        "param": str((args.output_dir / "seedvr2_frontend.ncnn.param").relative_to(args.output_dir.parent)),
        "bin": str((args.output_dir / "seedvr2_frontend.ncnn.bin").relative_to(args.output_dir.parent)),
        "input_order": ["patchified_video", "text_embedding", "timestep_embedding"],
        "output_order": [
            "video", "text", "attention_shift", "attention_scale", "attention_gate",
            "mlp_shift", "mlp_scale", "mlp_gate"
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    del traced, model
    gc.collect()
    for suffix in (".pt", ".pnnx.param", ".pnnx.bin", ".pnnx.onnx"):
        (args.output_dir / f"seedvr2_frontend{suffix}").unlink(missing_ok=True)
    for suffix in ("_pnnx.py", "_ncnn.py"):
        (args.output_dir / f"seedvr2_frontend{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
