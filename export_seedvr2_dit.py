"""Export the complete SeedVR2-3B DiT body as one NCNN graph."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DIM = 2560
HEADS = 20
HEAD_DIM = 128
INNER_DIM = HEADS * HEAD_DIM
HIDDEN_DIM = 6912
NUM_BLOCKS = 32
MM_BLOCKS = 10
EPSILON = 1e-5


def parameter(value: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(value, requires_grad=False)


def linear_from_state(state: dict[str, torch.Tensor], prefix: str) -> nn.Linear:
    weight = state[f"{prefix}.weight"]
    bias = state.get(f"{prefix}.bias")
    layer = nn.Linear(
        weight.shape[1], weight.shape[0], bias=bias is not None, device="meta"
    )
    layer.weight = parameter(weight)
    if bias is not None:
        layer.bias = parameter(bias)
    return layer


def sliced_qkv_state(
    state: dict[str, torch.Tensor], prefix: str, part: int
) -> dict[str, torch.Tensor]:
    """Slice part 0/1/2 (q/k/v) out of a fused [3*INNER_DIM, DIM] QKV state."""
    weight = state[f"{prefix}.weight"]
    if weight.shape[0] != 3 * INNER_DIM:
        raise ValueError(f"{prefix}.weight is not a fused QKV: {weight.shape}")
    result = {
        f"{prefix}.weight": weight[part * INNER_DIM:(part + 1) * INNER_DIM].contiguous()
    }
    bias = state.get(f"{prefix}.bias")
    if bias is not None:
        result[f"{prefix}.bias"] = bias[part * INNER_DIM:(part + 1) * INNER_DIM].contiguous()
    return result


def sliced_rows_state(
    state: dict[str, torch.Tensor], prefix: str, part: int, parts: int = 2
) -> dict[str, torch.Tensor]:
    """Slice a row range out of a [rows, cols] projection (gate/up split)."""
    weight = state[f"{prefix}.weight"]
    rows = weight.shape[0] // parts
    result = {
        f"{prefix}.weight": weight[part * rows:(part + 1) * rows].contiguous()
    }
    bias = state.get(f"{prefix}.bias")
    if bias is not None:
        result[f"{prefix}.bias"] = bias[part * rows:(part + 1) * rows].contiguous()
    return result


def sliced_cols_state(
    state: dict[str, torch.Tensor], prefix: str, part: int, parts: int = 2
) -> dict[str, torch.Tensor]:
    """Slice an input-column range out of a [rows, cols] projection.

    Used to split the down projection into a sum of partial Gemms so no
    intermediate activation exceeds the driver buffer limit. The bias is kept
    only on part 0 so the summed result matches the fused projection.
    """
    weight = state[f"{prefix}.weight"]
    cols = weight.shape[1] // parts
    result = {
        f"{prefix}.weight": weight[:, part * cols:(part + 1) * cols].contiguous()
    }
    bias = state.get(f"{prefix}.bias")
    if bias is not None and part == 0:
        result[f"{prefix}.bias"] = bias.contiguous()
    return result


class SharedLinear(nn.Module):
    """One serialized projection applied independently to two token streams."""

    def __init__(self, state: dict[str, torch.Tensor], prefix: str) -> None:
        super().__init__()
        self.register_buffer("a_weight", state[f"{prefix}.weight"])
        bias = state.get(f"{prefix}.bias")
        if bias is not None:
            self.register_buffer("b_bias", bias)

    def forward(
        self, video: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias = getattr(self, "b_bias", None)
        return (
            F.linear(video, self.a_weight, bias),
            F.linear(text, self.a_weight, bias),
        )


class RMSNorm(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = value.float().square().mean(-1, keepdim=True)
        return value * torch.rsqrt(variance + EPSILON).to(value.dtype)


class SwiGLU(nn.Module):
    """SwiGLU with column-split projections.

    gate/up are split along output rows and the down projection along input
    columns, so the largest intermediate is [tokens, HIDDEN_DIM//2] instead of
    [tokens, HIDDEN_DIM]. The math is identical to the fused projections.
    """

    def __init__(self, state: dict[str, torch.Tensor], prefix: str) -> None:
        super().__init__()
        self.gate_a = linear_from_state(sliced_rows_state(state, f"{prefix}.proj_in_gate", 0), f"{prefix}.proj_in_gate")
        self.gate_b = linear_from_state(sliced_rows_state(state, f"{prefix}.proj_in_gate", 1), f"{prefix}.proj_in_gate")
        self.up_a = linear_from_state(sliced_rows_state(state, f"{prefix}.proj_in", 0), f"{prefix}.proj_in")
        self.up_b = linear_from_state(sliced_rows_state(state, f"{prefix}.proj_in", 1), f"{prefix}.proj_in")
        self.down_a = linear_from_state(sliced_cols_state(state, f"{prefix}.proj_out", 0), f"{prefix}.proj_out")
        self.down_b = linear_from_state(sliced_cols_state(state, f"{prefix}.proj_out", 1), f"{prefix}.proj_out")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden_a = F.silu(self.gate_a(value)) * self.up_a(value)
        hidden_b = F.silu(self.gate_b(value)) * self.up_b(value)
        return self.down_a(hidden_a) + self.down_b(hidden_b)


class SharedSwiGLU(nn.Module):
    """Two-stream SwiGLU with column-split projections (see SwiGLU)."""

    def __init__(self, state: dict[str, torch.Tensor], prefix: str) -> None:
        super().__init__()
        self.gate_a = SharedLinear(sliced_rows_state(state, f"{prefix}.proj_in_gate", 0), f"{prefix}.proj_in_gate")
        self.gate_b = SharedLinear(sliced_rows_state(state, f"{prefix}.proj_in_gate", 1), f"{prefix}.proj_in_gate")
        self.up_a = SharedLinear(sliced_rows_state(state, f"{prefix}.proj_in", 0), f"{prefix}.proj_in")
        self.up_b = SharedLinear(sliced_rows_state(state, f"{prefix}.proj_in", 1), f"{prefix}.proj_in")
        self.down_a = SharedLinear(sliced_cols_state(state, f"{prefix}.proj_out", 0), f"{prefix}.proj_out")
        self.down_b = SharedLinear(sliced_cols_state(state, f"{prefix}.proj_out", 1), f"{prefix}.proj_out")

    def forward(
        self, video: torch.Tensor, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_gate_a, text_gate_a = self.gate_a(video, text)
        video_gate_b, text_gate_b = self.gate_b(video, text)
        video_up_a, text_up_a = self.up_a(video, text)
        video_up_b, text_up_b = self.up_b(video, text)
        video_hidden_a = F.silu(video_gate_a) * video_up_a
        video_hidden_b = F.silu(video_gate_b) * video_up_b
        text_hidden_a = F.silu(text_gate_a) * text_up_a
        text_hidden_b = F.silu(text_gate_b) * text_up_b
        video_down_a, text_down_a = self.down_a(video_hidden_a, text_hidden_a)
        video_down_b, text_down_b = self.down_b(video_hidden_b, text_hidden_b)
        return video_down_a + video_down_b, text_down_a + text_down_b


class AdaptiveWindowAttention(nn.Module):
    """PNNX opaque module. C++ supplies the actual CPU/Vulkan implementation."""

    def __init__(
        self,
        state: dict[str, torch.Tensor],
        prefix: str,
        shared: bool,
        fp16: bool = False,
    ) -> None:
        super().__init__()
        self.export_placeholder = False
        branch = "all" if shared else "vid"
        text_branch = "all" if shared else "txt"
        cast = lambda value: value.half().float() if fp16 else value
        # PNNX stores module attributes in lexical name order. The prefixes make
        # the NCNN ModelBin ABI explicit and stable.
        self.register_buffer(
            "a_video_norm_q", cast(state[f"{prefix}.norm_q.{branch}.weight"])
        )
        self.register_buffer(
            "b_video_norm_k", cast(state[f"{prefix}.norm_k.{branch}.weight"])
        )
        if not shared:
            self.register_buffer(
                "c_text_norm_q", cast(state[f"{prefix}.norm_q.{text_branch}.weight"])
            )
            self.register_buffer(
                "d_text_norm_k", cast(state[f"{prefix}.norm_k.{text_branch}.weight"])
            )
        self.register_buffer("e_rope_freqs", cast(state[f"{prefix}.rope.rope.freqs"]))

    def forward(
        self,
        video_q: torch.Tensor,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        text_q: torch.Tensor,
        text_k: torch.Tensor,
        text_v: torch.Tensor,
        video_shape: torch.Tensor,
        text_shape: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.export_placeholder:
            weight_dependency = (
                self.a_video_norm_q.sum()
                + self.b_video_norm_k.sum()
                + self.e_rope_freqs.sum()
            )
            if hasattr(self, "c_text_norm_q"):
                weight_dependency = (
                    weight_dependency
                    + self.c_text_norm_q.sum()
                    + self.d_text_norm_k.sum()
                )
            dependency = (
                (video_shape.sum() + text_shape.sum()).to(video_q)
                + weight_dependency.to(video_q)
            ) * 0
            return (
                video_q + dependency + video_k * 0 + video_v * 0,
                text_q + dependency + text_k * 0 + text_v * 0,
            )

        # The exporter traces batch one. Runtime batching is implemented by the
        # C++ custom layer and tested separately.
        t, h, w = (int(x) for x in video_shape[0].tolist())
        text_length = int(text_shape[0, 0])
        shared = not hasattr(self, "c_text_norm_q")
        text_norm_q = self.a_video_norm_q if shared else self.c_text_norm_q
        text_norm_k = self.b_video_norm_k if shared else self.d_text_norm_k

        video_q = self._rms_norm(
            video_q.reshape(-1, HEADS, HEAD_DIM), self.a_video_norm_q
        )
        video_k = self._rms_norm(
            video_k.reshape(-1, HEADS, HEAD_DIM), self.b_video_norm_k
        )
        video_v = video_v.reshape(-1, HEADS, HEAD_DIM)
        text_q = self._rms_norm(text_q.reshape(-1, HEADS, HEAD_DIM), text_norm_q)
        text_k = self._rms_norm(text_k.reshape(-1, HEADS, HEAD_DIM), text_norm_k)
        text_v = text_v.reshape(-1, HEADS, HEAD_DIM)
        text_positions = torch.arange(text_length, device=text_q.device)[:, None].repeat(1, 3)
        text_q = self._apply_rope(text_q, text_positions)
        text_k = self._apply_rope(text_k, text_positions)

        output_video = torch.empty_like(video_q)
        output_text = torch.zeros_like(text_q)
        index_grid = torch.arange(t * h * w, device=video_q.device).reshape(t, h, w)
        slices = self._window_slices((t, h, w), shifted=False)
        # During tracing each block is configured below with its actual parity.
        if getattr(self, "shifted", False):
            slices = self._window_slices((t, h, w), shifted=True)
        for window_slice in slices:
            indices = index_grid[window_slice].reshape(-1)
            window_shape = index_grid[window_slice].shape
            positions = torch.stack(
                torch.meshgrid(
                    *(torch.arange(size, device=video_q.device) for size in window_shape),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 3)
            positions[:, 0] += text_length
            query = torch.cat((self._apply_rope(video_q[indices], positions), text_q))
            key = torch.cat((self._apply_rope(video_k[indices], positions), text_k))
            value = torch.cat((video_v[indices], text_v))
            context = F.scaled_dot_product_attention(
                query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
            ).transpose(0, 1)
            output_video[indices] = context[: len(indices)]
            output_text += context[len(indices) :] / len(slices)
        return output_video.flatten(1), output_text.flatten(1)

    @staticmethod
    def _rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = value.float().square().mean(-1, keepdim=True)
        return value * torch.rsqrt(variance + EPSILON).to(value) * weight

    def _apply_rope(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        pairs = self.e_rope_freqs.numel()
        axis_width = pairs * 2
        for axis in range(3):
            for pair in range(pairs):
                d = axis * axis_width + pair * 2
                angle = positions[:, axis, None].to(value) * self.e_rope_freqs[pair]
                x0 = value[..., d]
                x1 = value[..., d + 1]
                result[..., d] = x0 * angle.cos() - x1 * angle.sin()
                result[..., d + 1] = x0 * angle.sin() + x1 * angle.cos()
        return result

    @staticmethod
    def _window_slices(shape: tuple[int, int, int], shifted: bool) -> list[tuple[slice, slice, slice]]:
        t, h, w = shape
        scale = math.sqrt(3600 / (h * w))
        resized_h, resized_w = round(h * scale), round(w * scale)
        wt = math.ceil(min(t, 30) / 4)
        wh = math.ceil(resized_h / 3)
        ww = math.ceil(resized_w / 3)
        if not shifted:
            return [
                (slice(it * wt, min((it + 1) * wt, t)),
                 slice(ih * wh, min((ih + 1) * wh, h)),
                 slice(iw * ww, min((iw + 1) * ww, w)))
                for iw in range(math.ceil(w / ww))
                for ih in range(math.ceil(h / wh))
                for it in range(math.ceil(t / wt))
            ]
        st, sh, sw = (0.5 if wt < t else 0, 0.5 if wh < h else 0, 0.5 if ww < w else 0)
        counts = [
            math.ceil((extent - shift) / window) + 1 if shift else 1
            for extent, shift, window in ((t, st, wt), (h, sh, wh), (w, sw, ww))
        ]
        result = []
        for iw in range(counts[2]):
            w0, w1 = max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)
            for ih in range(counts[1]):
                h0, h1 = max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)
                for it in range(counts[0]):
                    t0, t1 = max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)
                    if t1 > t0 and h1 > h0 and w1 > w0:
                        result.append((slice(t0, t1), slice(h0, h1), slice(w0, w1)))
        return result


class SeedVR2Block(nn.Module):
    def __init__(
        self, state: dict[str, torch.Tensor], index: int, fp16: bool = False
    ) -> None:
        super().__init__()
        self.index = index
        self.shared = index >= MM_BLOCKS
        self.video_only_mlp = index == NUM_BLOCKS - 1
        block = f"blocks.{index}"
        branch = "all" if self.shared else "vid"
        text_branch = "all" if self.shared else "txt"

        self.attention_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        qkv_prefix = f"{block}.attn.proj_qkv.{branch}"
        text_qkv_prefix = f"{block}.attn.proj_qkv.{text_branch}"
        if self.shared:
            self.shared_q = SharedLinear(
                sliced_qkv_state(state, qkv_prefix, 0), qkv_prefix
            )
            self.shared_k = SharedLinear(
                sliced_qkv_state(state, qkv_prefix, 1), qkv_prefix
            )
            self.shared_v = SharedLinear(
                sliced_qkv_state(state, qkv_prefix, 2), qkv_prefix
            )
        else:
            self.video_q = linear_from_state(sliced_qkv_state(state, qkv_prefix, 0), qkv_prefix)
            self.video_k = linear_from_state(sliced_qkv_state(state, qkv_prefix, 1), qkv_prefix)
            self.video_v = linear_from_state(sliced_qkv_state(state, qkv_prefix, 2), qkv_prefix)
            self.text_q = linear_from_state(sliced_qkv_state(state, text_qkv_prefix, 0), text_qkv_prefix)
            self.text_k = linear_from_state(sliced_qkv_state(state, text_qkv_prefix, 1), text_qkv_prefix)
            self.text_v = linear_from_state(sliced_qkv_state(state, text_qkv_prefix, 2), text_qkv_prefix)
        self.attention = AdaptiveWindowAttention(
            state, f"{block}.attn", self.shared, fp16
        )
        self.attention.shifted = index % 2 == 1
        if self.shared:
            self.shared_out = SharedLinear(
                state, f"{block}.attn.proj_out.{branch}"
            )
        else:
            self.video_out = linear_from_state(
                state, f"{block}.attn.proj_out.{branch}"
            )
            self.text_out = linear_from_state(
                state, f"{block}.attn.proj_out.{text_branch}"
            )
        if self.shared and not self.video_only_mlp:
            self.shared_mlp = SharedSwiGLU(state, f"{block}.mlp.{branch}")
            self.video_mlp = None
            self.text_mlp = None
        else:
            self.video_mlp = SwiGLU(state, f"{block}.mlp.{branch}")
            self.text_mlp = None if self.video_only_mlp else SwiGLU(
                state, f"{block}.mlp.{text_branch}"
            )

        ada_prefix = f"{block}.ada.{branch}"
        text_ada_prefix = f"{block}.ada.{text_branch}"
        for stage in ("attn", "mlp"):
            for kind in ("shift", "scale", "gate"):
                setattr(
                    self,
                    f"video_{stage}_{kind}",
                    parameter(state[f"{ada_prefix}.{stage}_{kind}"]),
                )
                if not self.shared and not (self.video_only_mlp and stage == "mlp"):
                    setattr(
                        self,
                        f"text_{stage}_{kind}",
                        parameter(state[f"{text_ada_prefix}.{stage}_{kind}"]),
                    )

    @staticmethod
    def modulate_in(
        value: torch.Tensor,
        dynamic_shift: torch.Tensor,
        dynamic_scale: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return value * (dynamic_scale + scale) + dynamic_shift + shift

    @staticmethod
    def modulate_out(
        value: torch.Tensor, dynamic_gate: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        return value * (dynamic_gate + gate)

    def forward(
        self,
        video: torch.Tensor,
        text: torch.Tensor,
        video_shape: torch.Tensor,
        text_shape: torch.Tensor,
        attention_shift: torch.Tensor,
        attention_scale: torch.Tensor,
        attention_gate: torch.Tensor,
        mlp_shift: torch.Tensor,
        mlp_scale: torch.Tensor,
        mlp_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_attention = self.modulate_in(
            self.attention_norm(video), attention_shift, attention_scale,
            self.video_attn_shift, self.video_attn_scale
        )
        text_attention = self.attention_norm(text)
        if not self.video_only_mlp:
            static_text_attn_shift = (
                self.video_attn_shift if self.shared else self.text_attn_shift
            )
            static_text_attn_scale = (
                self.video_attn_scale if self.shared else self.text_attn_scale
            )
            text_attention = self.modulate_in(
                text_attention, attention_shift, attention_scale,
                static_text_attn_shift, static_text_attn_scale
        )
        if self.shared:
            video_q, text_q = self.shared_q(video_attention, text_attention)
            video_k, text_k = self.shared_k(video_attention, text_attention)
            video_v, text_v = self.shared_v(video_attention, text_attention)
        else:
            video_q = self.video_q(video_attention)
            video_k = self.video_k(video_attention)
            video_v = self.video_v(video_attention)
            text_q = self.text_q(text_attention)
            text_k = self.text_k(text_attention)
            text_v = self.text_v(text_attention)
        video_context, text_context = self.attention(
            video_q, video_k, video_v, text_q, text_k, text_v,
            video_shape, text_shape
        )
        if self.shared:
            video_context, text_context = self.shared_out(
                video_context, text_context
            )
        else:
            video_context = self.video_out(video_context)
            text_context = self.text_out(text_context)
        video = video + self.modulate_out(
            video_context, attention_gate, self.video_attn_gate
        )
        if not self.video_only_mlp:
            static_text_attn_gate = (
                self.video_attn_gate if self.shared else self.text_attn_gate
            )
            text_context = self.modulate_out(
                text_context, attention_gate, static_text_attn_gate
            )
        text = text + text_context

        video_hidden = self.modulate_in(
            self.mlp_norm(video), mlp_shift, mlp_scale,
            self.video_mlp_shift, self.video_mlp_scale
        )
        if self.video_only_mlp:
            video_mlp = self.video_mlp(video_hidden)
            text_mlp = None
        else:
            static_text_mlp_shift = (
                self.video_mlp_shift if self.shared else self.text_mlp_shift
            )
            static_text_mlp_scale = (
                self.video_mlp_scale if self.shared else self.text_mlp_scale
            )
            text_hidden = self.modulate_in(
                self.mlp_norm(text), mlp_shift, mlp_scale,
                static_text_mlp_shift, static_text_mlp_scale
            )
            if self.shared:
                video_mlp, text_mlp = self.shared_mlp(
                    video_hidden, text_hidden
                )
            else:
                video_mlp = self.video_mlp(video_hidden)
                text_mlp = self.text_mlp(text_hidden)
        video = video + self.modulate_out(
            video_mlp, mlp_gate, self.video_mlp_gate
        )
        if text_mlp is not None:
            static_text_mlp_gate = (
                self.video_mlp_gate if self.shared else self.text_mlp_gate
            )
            text = text + self.modulate_out(
                text_mlp, mlp_gate, static_text_mlp_gate
            )
        else:
            # Official vid_only wrappers pass text through before the residual.
            text = text + text
        return video, text


class SeedVR2BlockGraph(nn.Module):
    """All 32 DiT blocks exported as one NCNN graph."""

    def __init__(self, state: dict[str, torch.Tensor], fp16: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SeedVR2Block(state, index, fp16) for index in range(NUM_BLOCKS)
        )

    def forward(
        self,
        video: torch.Tensor,
        text: torch.Tensor,
        video_shape: torch.Tensor,
        text_shape: torch.Tensor,
        attention_shift: torch.Tensor,
        attention_scale: torch.Tensor,
        attention_gate: torch.Tensor,
        mlp_shift: torch.Tensor,
        mlp_scale: torch.Tensor,
        mlp_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            video, text = block(
                video, text, video_shape, text_shape,
                attention_shift, attention_scale, attention_gate,
                mlp_shift, mlp_scale, mlp_gate,
            )
        return video, text


def parse_blocks(value: str) -> list[int]:
    if value == "all":
        return list(range(NUM_BLOCKS))
    result: set[int] = set()
    for item in value.split(","):
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(item))
    if not result or min(result) < 0 or max(result) >= NUM_BLOCKS:
        raise argparse.ArgumentTypeError("blocks must be in [0, 31]")
    return sorted(result)


def patch_custom_layer_params(
    pnnx_path: Path, ncnn_path: Path, block_indices: list[int]
) -> None:
    pnnx_lines = pnnx_path.read_text().splitlines()
    shared_shapes: dict[str, tuple[int, int, bool]] = {}
    for line in pnnx_lines:
        if not line.startswith("SharedLinear "):
            continue
        fields = line.split()
        match = re.search(r"@a_weight=\((\d+),(\d+)\)", line)
        if not match:
            raise RuntimeError(f"missing SharedLinear weight shape: {line}")
        shared_shapes[fields[1]] = (
            int(match.group(1)), int(match.group(2)), "@b_bias=" in line
        )

    lines = ncnn_path.read_text().splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith("AdaptiveWindowAttention ")]
    if len(matches) != len(block_indices):
        raise RuntimeError(
            f"expected {len(block_indices)} AdaptiveWindowAttention layers in "
            f"{ncnn_path}, got {len(matches)}"
        )
    for match, block_index in zip(matches, block_indices):
        fields = lines[match].split()
        input_count = int(fields[2])
        output_count = int(fields[3])
        if input_count != 8 or output_count != 2:
            raise RuntimeError(
                f"unexpected custom layer arity in {ncnn_path}: {fields[:4]}"
            )
        inputs = fields[4:12]
        outputs = fields[12:14]
        params = fields[14:]
        # Torch JIT orders scalar shape dependencies before projected tensors.
        # The C++ operator ABI is video q/k/v, text q/k/v, then shapes.
        inputs = [
            inputs[2], inputs[3], inputs[4],
            inputs[5], inputs[6], inputs[7],
            inputs[0], inputs[1],
        ]
        line = " ".join(fields[:4] + inputs + outputs + params)
        line += f" 0={HEADS} 1={HEAD_DIM} 2=4 3=3 4=3 5={int(block_index % 2 == 1)}"
        line += f" 6={EPSILON} 7=21 8={int(block_index >= MM_BLOCKS)} 9=0 10=1"
        lines[match] = line

    patched_shared = 0
    for index, line in enumerate(lines):
        if not line.startswith("SharedLinear "):
            continue
        fields = line.split()
        name = fields[1]
        if name not in shared_shapes:
            raise RuntimeError(f"missing PNNX metadata for SharedLinear {name}")
        num_output, num_input, bias = shared_shapes[name]
        lines[index] = line + f" 0={num_output} 1={num_input} 2={int(bias)}"
        patched_shared += 1
    if patched_shared != len(shared_shapes):
        raise RuntimeError(
            f"SharedLinear patch count mismatch: {patched_shared} != {len(shared_shapes)}"
        )
    ncnn_path.write_text("\n".join(lines) + "\n")


def verify_attribute_order(path: Path, block_indices: list[int]) -> None:
    attention_lines = [
        line for line in path.read_text().splitlines()
        if line.startswith("AdaptiveWindowAttention ")
    ]
    if len(attention_lines) != len(block_indices):
        raise RuntimeError(
            f"attention layer count mismatch in {path}: "
            f"{len(attention_lines)} != {len(block_indices)}"
        )
    for line, block_index in zip(attention_lines, block_indices):
        attributes = re.findall(r"@([A-Za-z0-9_]+)=", line)
        expected = ["a_video_norm_q", "b_video_norm_k"]
        if block_index < MM_BLOCKS:
            expected.extend(("c_text_norm_q", "d_text_norm_k"))
        expected.append("e_rope_freqs")
        if attributes != expected:
            raise RuntimeError(f"custom weight ABI mismatch: {attributes} != {expected}")

    for shared_line in (
        line for line in path.read_text().splitlines()
        if line.startswith("SharedLinear ")
    ):
        attributes = re.findall(r"@([A-Za-z0-9_]+)=", shared_line)
        if attributes not in (["a_weight"], ["a_weight", "b_bias"]):
            raise RuntimeError(
                f"SharedLinear weight ABI mismatch: {attributes}"
            )


def verify_projection_dedup(path: Path, block_indices: list[int]) -> None:
    lines = path.read_text().splitlines()
    gemms = sum(line.startswith("Gemm ") for line in lines)
    expected_gemms = sum(
        6 if block_index >= MM_BLOCKS and block_index == NUM_BLOCKS - 1
        else 0 if block_index >= MM_BLOCKS
        else 20
        for block_index in block_indices
    )
    if gemms != expected_gemms:
        raise RuntimeError(
            f"unexpected projection count in {path}: {gemms} != {expected_gemms}"
        )
    shared_linears = sum(line.startswith("SharedLinear ") for line in lines)
    expected_shared_linears = sum(
        4 if block_index == NUM_BLOCKS - 1 else 10
        for block_index in block_indices
        if block_index >= MM_BLOCKS
    )
    if shared_linears != expected_shared_linears:
        raise RuntimeError(
            f"unexpected SharedLinear count in {path}: "
            f"{shared_linears} != {expected_shared_linears}"
        )
    concats = sum(line.startswith("Concat ") for line in lines)
    crops = sum(line.startswith("Crop ") for line in lines)
    if any(block_index >= MM_BLOCKS for block_index in block_indices) and (concats, crops) != (0, 0):
        raise RuntimeError(
            f"shared projection must not allocate concat/crop buffers in {path}: "
            f"concat/crop={(concats, crops)}"
        )


def export_block_graph(
    state: dict[str, torch.Tensor], pnnx: Path, output_dir: Path,
    fp16: bool,
) -> dict[str, object]:
    block_indices = list(range(NUM_BLOCKS))
    block_dir = output_dir / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)
    model = SeedVR2BlockGraph(state, fp16).eval()
    for block in model.blocks:
        block.attention.export_placeholder = True
    example = (
        torch.zeros(8, DIM),
        torch.zeros(4, DIM),
        torch.tensor([[1.0, 2.0, 4.0]]),
        torch.tensor([[4.0]]),
        *(torch.zeros(1, DIM) for _ in range(6)),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    traced_path = block_dir / "seedvr2_blocks.pt"
    traced.save(traced_path)

    command = [
        str(pnnx),
        str(traced_path),
        "inputshape=[8,2560],[4,2560],[1,3],[1,1]," + ",".join(["[1,2560]"] * 6),
        "inputshape2=[12,2560],[5,2560],[1,3],[1,1]," + ",".join(["[1,2560]"] * 6),
        "moduleop=AdaptiveWindowAttention,SharedLinear",
        f"fp16={int(fp16)}",
    ]
    subprocess.run(command, cwd=block_dir, check=True)
    stem = traced_path.stem
    pnnx_param = block_dir / f"{stem}.pnnx.param"
    ncnn_param = block_dir / f"{stem}.ncnn.param"
    ncnn_bin = block_dir / f"{stem}.ncnn.bin"
    verify_attribute_order(pnnx_param, block_indices)
    patch_custom_layer_params(pnnx_param, ncnn_param, block_indices)
    verify_projection_dedup(ncnn_param, block_indices)
    result = {
        "blocks": block_indices,
        "param": str(ncnn_param.relative_to(output_dir)),
        "bin": str(ncnn_bin.relative_to(output_dir)),
        "bin_bytes": ncnn_bin.stat().st_size,
    }
    del traced, model
    gc.collect()
    for intermediate in (
        traced_path,
        block_dir / f"{stem}.pnnx.bin",
        block_dir / f"{stem}.pnnx.onnx",
        block_dir / f"{stem}_pnnx.py",
        block_dir / f"{stem}_ncnn.py",
        pnnx_param,
    ):
        intermediate.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true",
                        help="store NCNN weights in FP16 (default: FP32)")
    parser.add_argument("--blocks", type=parse_blocks, default=parse_blocks("all"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "format": "seedvr2-ncnn-dit-graph-v4",
        "source": "ByteDance-Seed/SeedVR2-3B",
        "architecture": {
            "dim": DIM,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "hidden_dim": HIDDEN_DIM,
            "blocks": NUM_BLOCKS,
            "multimodal_blocks": MM_BLOCKS,
            "window_count": [4, 3, 3],
            "patch_size": [1, 2, 2],
            "split_qkv": True,
            "split_mlp": True,
            "block_graph": True,
        },
        "block_graph": None,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        old_format = manifest.get("format")
        if old_format != "seedvr2-ncnn-dit-graph-v4":
            raise RuntimeError(
                f"cannot mix {old_format!r} and graph export; "
                "export graph to a fresh --output-dir"
            )
    if args.blocks != list(range(NUM_BLOCKS)):
        parser.error("the graph exporter requires --blocks all")
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    print(f"exporting all {NUM_BLOCKS} DiT blocks as one graph", flush=True)
    manifest["block_graph"] = export_block_graph(
        state, args.pnnx, args.output_dir, args.fp16
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
