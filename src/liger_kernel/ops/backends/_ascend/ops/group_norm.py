import os

os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt

from liger_kernel.ops.utils import ensure_contiguous


def _compute_batch_block_size(tile_size: int) -> int:
    return max(1, min(8, 4096 // tile_size))


def _group_norm_backward_spatial_block_size(hidden_size_per_channel: int) -> int:
    return min(1024, triton.next_power_of_2(hidden_size_per_channel))


def _group_norm_backward_batch_block_size(hidden_size_per_channel: int) -> int:
    block_h = min(1024, triton.next_power_of_2(hidden_size_per_channel))
    return _compute_batch_block_size(block_h)


def _group_norm_forward_single_task_block_sizes(
    hidden_size: int,
    channels_per_group: int,
    element_size: int,
) -> tuple[int, int]:
    hidden_size_per_channel = hidden_size // channels_per_group
    required = 32 // element_size
    block_h = max(min(1024, triton.next_power_of_2(hidden_size_per_channel)), required, 32)
    return block_h, triton.next_power_of_2(channels_per_group)


@triton.jit
def _group_norm_forward_kernel(
    Y_ptr,
    X_ptr,
    Mean_ptr,
    RSTD_ptr,
    W_ptr,
    B_ptr,
    num_groups,
    num_channels,
    spatial_size: tl.constexpr,
    channels_per_group: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    BLOCK_CH: tl.constexpr,
):
    pid_task = tl.program_id(0)
    pid_batch = pid_task // num_groups
    pid_group = pid_task % num_groups
    group_ch_start = pid_group * channels_per_group

    ch_offsets = tl.arange(0, BLOCK_CH)
    h_offsets = tl.arange(0, BLOCK_H)
    ch_mask = ch_offsets < channels_per_group
    h_mask = h_offsets < spatial_size
    mask = ch_mask[:, None] & h_mask[None, :]

    batch_stride = num_channels * spatial_size
    ch_stride = spatial_size
    ptrs = pid_batch * batch_stride + (group_ch_start + ch_offsets[:, None]) * ch_stride + h_offsets[None, :]

    X_vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
    hidden_size = channels_per_group * spatial_size
    sum_acc = tl.sum(X_vals)
    sum_sq_acc = tl.sum(X_vals * X_vals)

    mean = sum_acc / hidden_size
    variance = tl.maximum(sum_sq_acc / hidden_size - mean * mean, 0.0)
    rstd = rsqrt(variance + eps)

    W_vals = tl.load(W_ptr + group_ch_start + ch_offsets, mask=ch_mask, other=1.0).to(tl.float32)
    B_vals = tl.load(B_ptr + group_ch_start + ch_offsets, mask=ch_mask, other=0.0).to(tl.float32)
    Y_vals = (X_vals - mean) * rstd
    Y_vals = Y_vals * W_vals[:, None] + B_vals[:, None]
    tl.store(Y_ptr + ptrs, Y_vals.to(Y_ptr.dtype.element_ty), mask=mask, cache_modifier=".cg")

    tl.store(Mean_ptr + pid_task, mean)
    tl.store(RSTD_ptr + pid_task, rstd)


@triton.jit
def _group_norm_backward_kernel(
    X_ptr,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    DX_ptr,
    DW_ptr,
    DB_ptr,
    DY_ptr,
    batch_size,
    num_groups,
    num_channels,
    spatial_size: tl.constexpr,
    channels_per_group: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    pid_group = tl.program_id(0)
    pid_batch_block = tl.program_id(1)
    group_ch_start = pid_group * channels_per_group

    batch_offsets = pid_batch_block * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < batch_size
    task_offsets = batch_offsets * num_groups + pid_group
    h_offsets = tl.arange(0, BLOCK_H)

    mean = tl.load(Mean_ptr + task_offsets, mask=batch_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + task_offsets, mask=batch_mask, other=0.0).to(tl.float32)

    hidden_size = channels_per_group * spatial_size
    inv_hidden_size = 1.0 / hidden_size

    c1 = tl.zeros([BLOCK_BATCH], dtype=tl.float32)
    c2 = tl.zeros([BLOCK_BATCH], dtype=tl.float32)

    batch_stride = num_channels * spatial_size
    ch_stride = spatial_size

    for local_ch in tl.range(0, channels_per_group):
        dW_acc = 0.0
        dB_acc = 0.0

        for start in tl.range(0, spatial_size, BLOCK_H):
            idx = start + h_offsets
            h_mask = idx < spatial_size
            mask = batch_mask[:, None] & h_mask[None, :]
            ptrs = batch_offsets[:, None] * batch_stride + (group_ch_start + local_ch) * ch_stride + idx[None, :]

            X_vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
            DY_vals = tl.load(DY_ptr + ptrs, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)

            x_hat = (X_vals - mean[:, None]) * rstd[:, None]

            dW_acc += tl.sum(mask * DY_vals * x_hat)
            dB_acc += tl.sum(mask * DY_vals)
            c1 += tl.sum(mask * x_hat * DY_vals, axis=1)
            c2 += tl.sum(mask * DY_vals, axis=1)

        ch_idx = group_ch_start + local_ch
        tl.atomic_add(DW_ptr + ch_idx, dW_acc)
        tl.atomic_add(DB_ptr + ch_idx, dB_acc)

    c1 *= inv_hidden_size
    c2 *= inv_hidden_size

    for local_ch in tl.range(0, channels_per_group):
        W = tl.load(W_ptr + group_ch_start + local_ch).to(tl.float32)

        for start in tl.range(0, spatial_size, BLOCK_H):
            idx = start + h_offsets
            h_mask = idx < spatial_size
            mask = batch_mask[:, None] & h_mask[None, :]
            ptrs = batch_offsets[:, None] * batch_stride + (group_ch_start + local_ch) * ch_stride + idx[None, :]

            X_vals = tl.load(X_ptr + ptrs, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
            DY_vals = tl.load(DY_ptr + ptrs, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
            x_hat = (X_vals - mean[:, None]) * rstd[:, None]
            DX_vals = (W * DY_vals - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(DX_ptr + ptrs, DX_vals.to(DX_ptr.dtype.element_ty), mask=mask)


def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    spatial_size = X.numel() // (batch_size * num_channels)

    Y = torch.empty(shape, dtype=X.dtype, device=X.device)
    total_tasks = batch_size * num_groups
    Mean = torch.empty(total_tasks, dtype=torch.float32, device=X.device)
    RSTD = torch.empty(total_tasks, dtype=torch.float32, device=X.device)

    block_h, block_ch = _group_norm_forward_single_task_block_sizes(
        channels_per_group * spatial_size, channels_per_group, X.element_size()
    )
    _group_norm_forward_kernel[(total_tasks,)](
        Y,
        X,
        Mean,
        RSTD,
        W,
        B,
        num_groups,
        num_channels,
        spatial_size,
        channels_per_group,
        eps,
        BLOCK_H=block_h,
        BLOCK_CH=block_ch,
    )
    return Y, X, Mean.view(batch_size, num_groups), RSTD.view(batch_size, num_groups)


def group_norm_backward(dY, X, W, Mean, RSTD, num_channels, num_groups, bias_dtype):
    shape = dY.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    spatial_size = dY.numel() // (batch_size * num_channels)

    DX = torch.empty(shape, dtype=dY.dtype, device=dY.device)
    DW = torch.zeros(num_channels, dtype=torch.float32, device=W.device)
    DB = torch.zeros(num_channels, dtype=torch.float32, device=W.device)

    block_h = _group_norm_backward_spatial_block_size(spatial_size)
    block_batch = _group_norm_backward_batch_block_size(spatial_size)
    num_batch_blocks = triton.cdiv(batch_size, block_batch)

    _group_norm_backward_kernel[(num_groups, num_batch_blocks)](
        X,
        W,
        Mean,
        RSTD,
        DX,
        DW,
        DB,
        dY,
        batch_size,
        num_groups,
        num_channels,
        spatial_size,
        channels_per_group,
        BLOCK_H=block_h,
        BLOCK_BATCH=block_batch,
    )

    return DX, DW.to(W.dtype), DB.to(bias_dtype)


class LigerGroupNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        X,
        affine_scaling_weight,
        affine_shifting_bias,
        num_channels,
        num_groups,
        eps,
    ):
        Y, X, Mean, RSTD = group_norm_forward(
            X,
            num_channels,
            num_groups,
            affine_scaling_weight,
            affine_shifting_bias,
            eps,
        )
        ctx.num_channels = num_channels
        ctx.num_groups = num_groups
        ctx.bias_dtype = affine_shifting_bias.dtype
        ctx.save_for_backward(X, affine_scaling_weight, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = group_norm_backward(
            dY,
            X,
            W,
            Mean,
            RSTD,
            ctx.num_channels,
            ctx.num_groups,
            ctx.bias_dtype,
        )
        return DX, DW, DB, None, None, None
