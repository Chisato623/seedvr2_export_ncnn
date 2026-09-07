"""Export SeedVR2 VAE with Conv3D decomposed into Vulkan-capable Conv2D ops."""

from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


def parse_ncnn_layer(line: str) -> tuple[list[str], list[str], list[str]]:
    fields = line.split()
    bottom_count = int(fields[2])
    top_count = int(fields[3])
    bottoms = fields[4 : 4 + bottom_count]
    tops = fields[4 + bottom_count : 4 + bottom_count + top_count]
    params = fields[4 + bottom_count + top_count :]
    return bottoms, tops, params


def ncnn_layer(
    layer_type: str,
    name: str,
    bottoms: list[str],
    tops: list[str],
    params: list[str],
) -> str:
    fields = [layer_type, name, str(len(bottoms)), str(len(tops))]
    fields.extend(bottoms)
    fields.extend(tops)
    fields.extend(params)
    return " ".join(fields) + "\n"


def fuse_dynamic_mid_attention(lines: list[str]) -> list[str]:
    """Restore PNNX MHA fusion lost when the temporal dimension is dynamic."""
    while True:
        sdpa_index = next(
            (i for i, line in enumerate(lines) if line.split()[:1] == ["SDPA"]),
            None,
        )
        if sdpa_index is None:
            break

        start = None
        for index in range(sdpa_index - 3, 1, -1):
            types = [line.split()[0] for line in lines[index : index + 4]]
            if types == ["Split", "Reshape", "Split", "GroupNorm"]:
                start = index
                break
        if start is None or lines[start - 1].split()[0] != "Reshape":
            raise RuntimeError("could not locate dynamic mid-attention prefix")
        end = sdpa_index + 9
        expected_suffix = [
            "Permute", "Reshape", "InnerProduct", "Permute", "Reshape",
            "Reshape", "Reshape", "BinaryOp", "Reshape",
        ]
        if [line.split()[0] for line in lines[sdpa_index + 1 : end + 1]] != expected_suffix:
            raise RuntimeError("unexpected dynamic mid-attention suffix")

        pre_fields = lines[start - 1].split()
        pre_bottoms, pre_tops, _ = parse_ncnn_layer(lines[start - 1])
        lines[start - 1] = ncnn_layer(
            "Reshape", pre_fields[1], pre_bottoms, pre_tops,
            ['12=1', '13=0', '6="1w,1h,512,1d"'],
        )

        outer_fields = lines[start].split()
        outer_bottoms, outer_tops, outer_params = parse_ncnn_layer(lines[start])
        if len(outer_tops) != 3:
            raise RuntimeError("unexpected dynamic mid-attention outer split")
        residual_blob, shape_blob, attention_blob = outer_tops

        flat_fields = lines[start + 1].split()
        _, flat_tops, _ = parse_ncnn_layer(lines[start + 1])
        gn_fields = lines[start + 3].split()
        _, gn_tops, gn_params = parse_ncnn_layer(lines[start + 3])
        pre_permute_fields = lines[start + 4].split()
        _, pre_permute_tops, pre_permute_params = parse_ncnn_layer(lines[start + 4])
        projection_fields = lines[sdpa_index + 3].split()
        _, projection_tops, _ = parse_ncnn_layer(lines[sdpa_index + 3])
        post_permute_fields = lines[sdpa_index + 4].split()
        _, post_permute_tops, post_permute_params = parse_ncnn_layer(
            lines[sdpa_index + 4]
        )
        spatial_fields = lines[sdpa_index + 5].split()
        _, spatial_tops, _ = parse_ncnn_layer(lines[sdpa_index + 5])
        add_fields = lines[sdpa_index + 8].split()
        _, add_tops, add_params = parse_ncnn_layer(lines[sdpa_index + 8])
        final_fields = lines[sdpa_index + 9].split()
        _, final_tops, _ = parse_ncnn_layer(lines[sdpa_index + 9])

        replacement = [
            ncnn_layer("Split", outer_fields[1], outer_bottoms, outer_tops, outer_params),
            ncnn_layer("Reshape", flat_fields[1], [attention_blob], flat_tops,
                       ["0=-1", "1=512"]),
            ncnn_layer("GroupNorm", gn_fields[1], flat_tops, gn_tops, gn_params),
            ncnn_layer("Permute", pre_permute_fields[1], gn_tops,
                       pre_permute_tops, pre_permute_params),
            ncnn_layer("SeedVR2ChunkedMHA", "dynamic_mid_attention",
                       pre_permute_tops, projection_tops,
                       ["0=512", "1=1", "2=262144", "3=512", "4=512"]),
            ncnn_layer("Permute", post_permute_fields[1], projection_tops,
                       post_permute_tops, post_permute_params),
            ncnn_layer("Reshape", spatial_fields[1],
                       [post_permute_tops[0], shape_blob], spatial_tops,
                       ['6="1w,1h,512"']),
            ncnn_layer("BinaryOp", add_fields[1],
                       [spatial_tops[0], residual_blob], add_tops, add_params),
            ncnn_layer("Reshape", final_fields[1],
                       [add_tops[0]], final_tops,
                       ["12=0", "13=1", '6="0w,0h,512,0n,1"']),
        ]
        lines[start : end + 1] = replacement
    return lines


def prune_dead_split_outputs(lines: list[str]) -> list[str]:
    """Drop Split aliases that are neither consumed nor model outputs."""
    consumed = {
        bottom
        for line in lines[2:]
        for bottom in parse_ncnn_layer(line)[0]
    }
    for index, line in enumerate(lines[2:], start=2):
        fields = line.split()
        if fields[0] != "Split":
            continue
        bottoms, tops, params = parse_ncnn_layer(line)
        retained = [top for top in tops if top in consumed or top == "out0"]
        if not retained:
            raise RuntimeError(f"Split {fields[1]} has no live output")
        if len(retained) != len(tops):
            lines[index] = ncnn_layer(
                fields[0], fields[1], bottoms, retained, params
            )
    return lines


def patch_ncnn_dynamic_param(path: Path) -> None:
    """Translate unsupported PNNX five-rank size expressions to NCNN axes."""
    text = path.read_text()
    text = (
        text.replace("size(@1,4)", "1w")
        .replace("size(@1,3)", "1h")
        .replace("size(@1,2)", "1d")
    )
    text = text.replace('6="0,0,', '6="0w,0h,')
    text = text.replace('6="0w,0h,0,', '6="0w,0h,0d,')
    text = re.sub(
        r"0=0 1=0 11=(-?\d+) 12=(-?\d+) 13=(-?\d+) 2=(-?\d+)",
        r'12=\2 13=\3 6="0w,0h,\1,\4"',
        text,
    )
    lines = text.splitlines(keepends=True)
    producers: dict[str, str] = {}
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) < 4:
            continue
        bottom_count = int(fields[2])
        top_count = int(fields[3])
        producer = producers.get(fields[4]) if bottom_count > 0 else None
        restores_four_rank_video = re.search(
            r'6="0w,0h,0d,\d+"', line
        ) is not None
        if (
            fields[0] == "Reshape"
            and bottom_count > 0
            and (
                producer == "Swish"
                or (producer == "Split" and restores_four_rank_video)
            )
        ):
            lines[index] = line.replace(
                '6="0w,0h,0d,', '6="0w,0h,0n,'
            )
        for top_index in range(top_count):
            top = fields[4 + bottom_count + top_index]
            producers[top] = fields[0]
    lines = fuse_dynamic_mid_attention(lines)
    for index, line in enumerate(lines):
        if line.split()[:1] != ["MultiHeadAttention"]:
            continue
        final_index = index + 4
        if final_index >= len(lines) or lines[final_index].split()[0] != "Reshape":
            raise RuntimeError("could not locate fused attention output reshape")
        final_fields = lines[final_index].split()
        final_bottoms, final_tops, _ = parse_ncnn_layer(lines[final_index])
        lines[final_index] = ncnn_layer(
            "Reshape", final_fields[1], [final_bottoms[0]], final_tops,
            ["12=0", "13=1", '6="0w,0h,512,0n,1"'],
        )
    lines = prune_dead_split_outputs(lines)
    layer_count = len(lines) - 2
    blob_count = sum(int(line.split()[3]) for line in lines[2:])
    lines[1] = f"{layer_count} {blob_count}\n"
    text = "".join(lines)
    if "size(@" in text:
        raise RuntimeError(f"unsupported dynamic shape expression remains in {path}")
    if "0=0 1=0" in text or '6="0,0,' in text:
        raise RuntimeError(f"ambiguous dynamic spatial zero remains in {path}")
    if '6="0w,0h,0,' in text:
        raise RuntimeError(f"ambiguous dynamic temporal zero remains in {path}")
    path.write_text(text)


class DecomposedConv3d(nn.Module):
    """Exact Conv3D as temporal slices of Conv2D, preserving causal head pad."""

    def __init__(self, source: nn.Conv3d, causal: bool) -> None:
        super().__init__()
        self.kernel_t = source.kernel_size[0]
        self.stride_t = source.stride[0]
        self.dilation_t = source.dilation[0]
        self.head_frames = int(getattr(source, "temporal_padding", 0)) * 2 if causal else 0
        self.register_buffer(
            "a_config",
            torch.tensor(
                [
                    source.out_channels,
                    source.in_channels,
                    source.kernel_size[2],
                    source.kernel_size[1],
                    source.kernel_size[0],
                    source.stride[0],
                    source.dilation[0],
                    self.head_frames,
                    source.stride[2],
                    source.stride[1],
                    source.dilation[2],
                    source.dilation[1],
                    source.padding[2],
                    source.padding[1],
                    source.groups,
                    int(source.bias is not None),
                ],
                dtype=torch.float32,
            ),
        )
        self.spatial = nn.ModuleList()
        for temporal_index in range(self.kernel_t):
            conv = nn.Conv2d(
                source.in_channels,
                source.out_channels,
                kernel_size=source.kernel_size[1:],
                stride=source.stride[1:],
                padding=source.padding[1:],
                dilation=source.dilation[1:],
                groups=source.groups,
                bias=False,
                device="meta",
            )
            conv.weight = nn.Parameter(
                source.weight[:, :, temporal_index].detach(), requires_grad=False
            )
            self.spatial.append(conv)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(source.bias.detach(), requires_grad=False)

    def forward(self, value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.head_frames:
            head = value[:, :, :1].repeat(1, 1, self.head_frames, 1, 1)
            value = torch.cat((head, value), dim=2)
        output_t = (
            value.shape[2] - self.dilation_t * (self.kernel_t - 1) - 1
        ) // self.stride_t + 1
        result = None
        batch = value.shape[0]
        for temporal_index, conv in enumerate(self.spatial):
            offset = temporal_index * self.dilation_t
            frames = value[
                :, :, offset : offset + output_t * self.stride_t : self.stride_t
            ]
            frames = frames.permute(0, 2, 1, 3, 4).reshape(
                batch * output_t, frames.shape[1], frames.shape[3], frames.shape[4]
            )
            current = conv(frames)
            result = current if result is None else result + current
        assert result is not None
        if self.bias is not None:
            result = result + self.bias.reshape(1, -1, 1, 1)
        result = result.reshape(
            batch, output_t, result.shape[1], result.shape[2], result.shape[3]
        )
        result = result.permute(0, 2, 1, 3, 4)
        return result + self.a_config.sum().to(result) * 0


def replace_conv3d(module: nn.Module, causal_type: type[nn.Module]) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d):
            setattr(module, name, DecomposedConv3d(child, isinstance(child, causal_type)))
        else:
            replace_conv3d(child, causal_type)


class ExportUpsample3D(nn.Module):
    """Stable PNNX boundary for SeedVR2's spatiotemporal pixel shuffle."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        if source.slicing or not source.use_conv or source.name != "conv":
            raise ValueError("unsupported SeedVR2 Upsample3D variant")
        self.register_buffer(
            "a_config",
            torch.tensor(
                [
                    source.channels,
                    source.temporal_ratio,
                    source.spatial_ratio,
                    int(source.temporal_up),
                ],
                dtype=torch.float32,
            ),
        )
        self.b_post_conv = source.conv
        self.c_upscale_conv = source.upscale_conv
        self.temporal_ratio = source.temporal_ratio
        self.spatial_ratio = source.spatial_ratio
        self.temporal_up = source.temporal_up

    def forward(self, value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        value = self.c_upscale_conv(value)
        batch, _, frames, height, width = value.shape
        channels = int(self.a_config[0].item())
        value = value.reshape(
            batch,
            self.spatial_ratio,
            self.spatial_ratio,
            self.temporal_ratio,
            channels,
            frames,
            height,
            width,
        )
        value = value.permute(0, 4, 5, 3, 6, 1, 7, 2).reshape(
            batch,
            channels,
            frames * self.temporal_ratio,
            height * self.spatial_ratio,
            width * self.spatial_ratio,
        )
        if self.temporal_up:
            value = torch.cat((value[:, :, :1], value[:, :, 2:]), dim=2)
        value = self.b_post_conv(value)
        return value + self.a_config.sum().to(value) * 0


def replace_upsample3d(module: nn.Module, upsample_type: type[nn.Module]) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, upsample_type):
            setattr(module, name, ExportUpsample3D(child))
        else:
            replace_upsample3d(child, upsample_type)


class ExportDownsample3D(nn.Module):
    """Stable boundary for SeedVR2's asymmetric spatial downsample pad."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        if not source.use_conv or source.padding != 0 or not source.spatial_down:
            raise ValueError("unsupported SeedVR2 Downsample3D variant")
        self.register_buffer(
            "a_config",
            torch.tensor([source.channels, 1, 1], dtype=torch.float32),
        )
        self.b_conv = source.conv

    def forward(self, value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        value = torch.nn.functional.pad(value, (0, 1, 0, 1), value=0)
        value = self.b_conv(value)
        return value + self.a_config.sum().to(value) * 0


def replace_downsample3d(module: nn.Module, downsample_type: type[nn.Module]) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, downsample_type) and child.spatial_down:
            setattr(module, name, ExportDownsample3D(child))
        else:
            replace_downsample3d(child, downsample_type)


class DecoderWrapper(nn.Module):
    def __init__(self, vae: nn.Module, memory_state) -> None:
        super().__init__()
        self.decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv
        self.memory_state = memory_state

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self.post_quant_conv is not None:
            latent = self.post_quant_conv(latent, memory_state=self.memory_state)
        return self.decoder(latent, memory_state=self.memory_state)


class EncoderWrapper(nn.Module):
    def __init__(self, vae: nn.Module, memory_state) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.memory_state = memory_state

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        moments = self.encoder(video, memory_state=self.memory_state)
        if self.quant_conv is not None:
            moments = self.quant_conv(moments, memory_state=self.memory_state)
        return moments


def export_model(
    model: nn.Module,
    name: str,
    example: torch.Tensor,
    secondary_example: torch.Tensor | None,
    pnnx: Path,
    output_dir: Path,
) -> None:
    model.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    traced_path = output_dir / f"{name}.pt"
    traced.save(traced_path)
    command = [
        str(pnnx),
        str(traced_path),
        "inputshape=" + str(list(example.shape)).replace(" ", ""),
        "moduleop=DecomposedConv3d,ExportUpsample3D,ExportDownsample3D",
        "fp16=1",
    ]
    if secondary_example is not None:
        command.append(
            "inputshape2="
            + str(list(secondary_example.shape)).replace(" ", "")
        )
    subprocess.run(
        command,
        cwd=output_dir,
        check=True,
    )
    if secondary_example is not None:
        patch_ncnn_dynamic_param(output_dir / f"{name}.ncnn.param")
    del traced
    gc.collect()
    for suffix in (".pt", ".pnnx.param", ".pnnx.bin", ".pnnx.onnx"):
        (output_dir / f"{name}{suffix}").unlink(missing_ok=True)
    for suffix in ("_pnnx.py", "_ncnn.py"):
        (output_dir / f"{name}{suffix}").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seedvr-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part", choices=("encoder", "decoder", "both"), default="both")
    parser.add_argument(
        "--dynamic-spatial",
        action="store_true",
        help="derive runtime spatial expressions from 32x32 and 64x64 examples",
    )
    parser.add_argument(
        "--dynamic-temporal",
        action="store_true",
        help="derive runtime temporal expressions for official VAE slices",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(args.seedvr_root)
    sys.path.insert(0, str(args.seedvr_root))
    from common.config import create_object, load_config
    from models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
    from models.video_vae_v3.modules.attn_video_vae import Downsample3D, Upsample3D
    from models.video_vae_v3.modules.types import MemoryState

    config = load_config("configs_3b/main.yaml")
    vae = create_object(config.vae.model).eval()
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    vae.load_state_dict(state)
    replace_conv3d(vae, InflatedCausalConv3d)
    replace_upsample3d(vae, Upsample3D)
    replace_downsample3d(vae, Downsample3D)

    dynamic_shape = args.dynamic_spatial or args.dynamic_temporal
    decoder_secondary = None
    encoder_secondary = None
    if dynamic_shape:
        decoder_secondary = torch.zeros(
            1,
            16,
            1 if args.dynamic_temporal else 2,
            8 if args.dynamic_spatial else 4,
            8 if args.dynamic_spatial else 4,
        )
        encoder_secondary = torch.zeros(
            1,
            3,
            4 if args.dynamic_temporal else 5,
            64 if args.dynamic_spatial else 32,
            64 if args.dynamic_spatial else 32,
        )

    if args.part in ("decoder", "both"):
        export_model(
            DecoderWrapper(vae, MemoryState.DISABLED),
            "seedvr2_vae_decoder",
            torch.zeros(1, 16, 2, 4, 4),
            decoder_secondary,
            args.pnnx,
            args.output_dir,
        )
    if args.part in ("encoder", "both"):
        export_model(
            EncoderWrapper(vae, MemoryState.DISABLED),
            "seedvr2_vae_encoder",
            torch.zeros(1, 3, 5, 32, 32),
            encoder_secondary,
            args.pnnx,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
