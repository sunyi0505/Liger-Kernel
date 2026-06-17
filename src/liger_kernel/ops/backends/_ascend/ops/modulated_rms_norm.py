"""
Ascend NPU implementation of modulated RMSNorm.

Fused kernel: ``y = (1 + scale) * RMSNorm(x) + shift``.

Extends the Ascend rms_norm kernels with AdaLN-style scale/shift modulation.
"""

import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import ensure_contiguous
from liger_kernel.ops.utils import get_npu_core_count
from liger_kernel.ops.utils import torch_to_triton_dtype

_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


def torch_dtype_to_triton(dtype):
    mapping = {
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
    }
    return mapping.get(dtype, tl.float32)


# -----------------------------------------------------------------------------
# Forward Kernel - No Tiling (for n_cols <= 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _modulated_rms_norm_forward_kernel_no_tiling(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    Scale_ptr,
    Scale_row_stride,
    Shift_ptr,
    Shift_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    has_shift: tl.constexpr,
    rows_per_modulation: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)

    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)

    for i in range(num_iterations):
        row_idx = i * grid_stride + pid * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]
        mod_row_idx = row_idx // rows_per_modulation

        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
        )
        Scale_rows = tl.load(
            Scale_ptr + mod_row_idx[:, None] * Scale_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
        )

        if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
            X_rows = X_rows.to(tl.float32)

        sum_squares = tl.sum(tl.where(block_mask, X_rows * X_rows, 0.0), axis=1)
        mean_squares = sum_squares / n_cols
        rstd_rows = rsqrt(mean_squares + eps)

        tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd_rows, mask=row_mask)

        X_rows = X_rows * rstd_rows[:, None]

        if casting_mode == _CASTING_MODE_LLAMA:
            X_rows = X_rows.to(X_DTYPE)

        if elementwise_affine:
            if casting_mode == _CASTING_MODE_GEMMA:
                Y_rows = X_rows * (offset + W_row.to(tl.float32)[None, :])
            else:
                Y_rows = X_rows * (offset + W_row[None, :])
        else:
            Y_rows = X_rows

        if casting_mode == _CASTING_MODE_GEMMA:
            Y_rows = Y_rows.to(X_DTYPE)

        Y_rows = Y_rows * (1.0 + Scale_rows)
        if has_shift:
            Shift_rows = tl.load(
                Shift_ptr + mod_row_idx[:, None] * Shift_row_stride + col_offsets[None, :],
                mask=block_mask,
                other=0.0,
            )
            Y_rows = Y_rows + Shift_rows

        tl.store(
            Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
            Y_rows,
            mask=block_mask,
        )


# -----------------------------------------------------------------------------
# Forward Kernel - With Tiling (for n_cols > 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _modulated_rms_norm_forward_kernel_tiled(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    Scale_ptr,
    Scale_row_stride,
    Shift_ptr,
    Shift_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    has_shift: tl.constexpr,
    rows_per_modulation: tl.constexpr,
    X_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(X_DTYPE)
        offset = offset.to(X_DTYPE)

    offsets = tl.arange(0, BLOCK_SIZE)

    for row_idx in tl.range(pid, n_rows, num_progs):
        mod_row_idx = row_idx // rows_per_modulation
        Y_row_ptr = Y_ptr + row_idx * Y_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride
        scale_row_ptr = Scale_ptr + mod_row_idx * Scale_row_stride
        shift_row_ptr = Shift_ptr + mod_row_idx * Shift_row_stride if has_shift else 0

        sum_square = 0.0
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                X_block = X_block.to(tl.float32)
            sum_square += tl.sum(X_block * X_block)

        mean_square = sum_square / n_cols
        rstd = rsqrt(mean_square + eps)
        tl.store(RSTD_row_ptr, rstd)

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
            Scale_block = tl.load(scale_row_ptr + col_offsets, mask=mask, other=0.0)

            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                X_block = X_block.to(tl.float32)

            X_block = X_block * rstd

            if casting_mode == _CASTING_MODE_LLAMA:
                X_block = X_block.to(X_DTYPE)

            if elementwise_affine:
                if casting_mode == _CASTING_MODE_GEMMA:
                    W_block = W_block.to(tl.float32)
                Y_block = X_block * (offset + W_block)
            else:
                Y_block = X_block

            if casting_mode == _CASTING_MODE_GEMMA:
                Y_block = Y_block.to(X_DTYPE)

            Y_block = Y_block * (1.0 + Scale_block)
            if has_shift:
                Shift_block = tl.load(shift_row_ptr + col_offsets, mask=mask, other=0.0)
                Y_block = Y_block + Shift_block

            tl.store(Y_row_ptr + col_offsets, Y_block, mask=mask)


# -----------------------------------------------------------------------------
# Backward Kernel - No Tiling (n_cols <= 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _modulated_rms_norm_backward_kernel_no_tiling(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    Scale_ptr,
    Scale_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dScale_ptr,
    dScale_row_stride,
    dShift_ptr,
    dShift_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    has_shift: tl.constexpr,
    rows_per_modulation: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_stride = num_progs * BLOCK_SIZE_M
    num_iterations = tl.cdiv(n_rows, grid_stride)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    row_offsets = tl.arange(0, BLOCK_SIZE_M)

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
        W_offset = W_row + offset

    for i in range(num_iterations):
        row_idx = i * grid_stride + pid * BLOCK_SIZE_M + row_offsets
        row_mask = row_idx < n_rows
        block_mask = row_mask[:, None] & col_mask[None, :]
        mod_row_idx = row_idx // rows_per_modulation

        dY_rows = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
        )
        X_rows = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
        )
        Scale_rows = tl.load(
            Scale_ptr + mod_row_idx[:, None] * Scale_row_stride + col_offsets[None, :],
            mask=block_mask,
            other=0.0,
        )
        rstd_rows = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, mask=row_mask, other=0.0)

        X_rows = X_rows.to(tl.float32)
        X_norm = X_rows * rstd_rows[:, None]
        Mod_rows = 1.0 + Scale_rows
        dRms_rows = dY_rows * Mod_rows

        if elementwise_affine:
            if casting_mode == _CASTING_MODE_LLAMA:
                m_rows = (dRms_rows * W_offset[None, :]).to(tl.float32)
                rms_output = X_norm.to(X_dtype) * W_offset[None, :]
            elif casting_mode == _CASTING_MODE_GEMMA:
                m_rows = dRms_rows.to(tl.float32) * W_offset[None, :]
                rms_output = (X_norm * W_offset[None, :]).to(X_dtype)
            else:
                m_rows = dRms_rows * W_offset[None, :]
                rms_output = X_norm * W_offset[None, :]
        else:
            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                m_rows = dRms_rows.to(tl.float32)
            else:
                m_rows = dRms_rows
            if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                rms_output = X_norm.to(X_dtype)
            else:
                rms_output = X_norm

        sum_m_X = tl.sum(tl.where(block_mask, m_rows * X_rows, 0.0), axis=1)
        correction_factors = -(1.0 / n_cols) * rstd_rows * rstd_rows * sum_m_X
        dX_rows = rstd_rows[:, None] * m_rows + rstd_rows[:, None] * correction_factors[:, None] * X_rows

        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX_rows.to(X_dtype),
            mask=block_mask,
        )

        dScale_rows = (dY_rows * rms_output).to(X_dtype)
        tl.store(
            dScale_ptr + row_idx[:, None] * dScale_row_stride + col_offsets[None, :],
            dScale_rows,
            mask=block_mask,
        )
        if has_shift:
            if rows_per_modulation == 1:
                tl.store(
                    dShift_ptr + mod_row_idx[:, None] * dShift_row_stride + col_offsets[None, :],
                    dY_rows,
                    mask=block_mask,
                )
            else:
                tl.store(
                    dShift_ptr + row_idx[:, None] * dShift_row_stride + col_offsets[None, :],
                    dY_rows,
                    mask=block_mask,
                )


# -----------------------------------------------------------------------------
# Backward Kernel - With Tiling (n_cols > 2048)
# -----------------------------------------------------------------------------


@triton.jit
def _modulated_rms_norm_backward_kernel_tiled(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    Scale_ptr,
    Scale_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dScale_ptr,
    dScale_row_stride,
    dShift_ptr,
    dShift_row_stride,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    has_shift: tl.constexpr,
    rows_per_modulation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_col_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)

    for row_idx in tl.range(pid, n_rows, num_progs):
        mod_row_idx = row_idx // rows_per_modulation
        dY_row_ptr = dY_ptr + row_idx * dY_row_stride
        dX_row_ptr = dX_ptr + row_idx * dX_row_stride
        X_row_ptr = X_ptr + row_idx * X_row_stride
        RSTD_row_ptr = RSTD_ptr + row_idx * RSTD_row_stride
        scale_row_ptr = Scale_ptr + mod_row_idx * Scale_row_stride
        dshift_row_ptr = dShift_ptr + mod_row_idx * dShift_row_stride if has_shift else 0

        rstd = tl.load(RSTD_row_ptr)

        sum_m_X = 0.0
        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
            Scale_block = tl.load(scale_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = X_block.to(tl.float32)
            Mod_block = 1.0 + Scale_block
            dRms_block = dY_block * Mod_block

            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
                W_offset = W_block + offset
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = (dRms_block * W_offset).to(tl.float32)
                elif casting_mode == _CASTING_MODE_GEMMA:
                    m = dRms_block.to(tl.float32) * W_offset
                else:
                    m = dRms_block * W_offset
            else:
                if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                    m = dRms_block.to(tl.float32)
                else:
                    m = dRms_block

            sum_m_X += tl.sum(m * X_block)

        correction_factor = -(1.0 / n_cols) * rstd * rstd * sum_m_X

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = col_start + offsets
            mask = col_offsets < n_cols

            dY_block = tl.load(dY_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
            Scale_block = tl.load(scale_row_ptr + col_offsets, mask=mask, other=0.0)
            X_block = X_block.to(tl.float32)
            X_norm = X_block * rstd
            Mod_block = 1.0 + Scale_block
            dRms_block = dY_block * Mod_block

            if elementwise_affine:
                W_block = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
                W_offset = W_block + offset
                if casting_mode == _CASTING_MODE_LLAMA:
                    m = (dRms_block * W_offset).to(tl.float32)
                    rms_output = X_norm.to(X_dtype) * W_offset
                elif casting_mode == _CASTING_MODE_GEMMA:
                    m = dRms_block.to(tl.float32) * W_offset
                    rms_output = (X_norm * W_offset).to(X_dtype)
                else:
                    m = dRms_block * W_offset
                    rms_output = X_norm * W_offset
            else:
                if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                    m = dRms_block.to(tl.float32)
                else:
                    m = dRms_block
                if casting_mode == _CASTING_MODE_LLAMA or casting_mode == _CASTING_MODE_GEMMA:
                    rms_output = X_norm.to(X_dtype)
                else:
                    rms_output = X_norm

            dX_block = rstd * m + rstd * correction_factor * X_block
            tl.store(dX_row_ptr + col_offsets, dX_block.to(X_dtype), mask=mask)

            dScale_block = (dY_block * rms_output).to(X_dtype)
            if rows_per_modulation == 1:
                dscale_row_ptr = dScale_ptr + mod_row_idx * dScale_row_stride
                tl.store(dscale_row_ptr + col_offsets, dScale_block, mask=mask)
            else:
                dscale_row_ptr = dScale_ptr + row_idx * dScale_row_stride
                tl.store(dscale_row_ptr + col_offsets, dScale_block, mask=mask)

            if has_shift:
                if rows_per_modulation == 1:
                    tl.store(dshift_row_ptr + col_offsets, dY_block, mask=mask)
                else:
                    tl.atomic_add(dshift_row_ptr + col_offsets, dY_block, sem="relaxed", mask=mask)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_optimal_block_size(n_cols, is_forward: bool):
    if n_cols <= 2048:
        return triton.next_power_of_2(n_cols)

    memory_multiplier = 8.0 if is_forward else 10.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9,
        dtype_size=4,
        memory_multiplier=memory_multiplier,
        shapes=((n_cols,),),
        tiling_dims=(0,),
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return max(2048, block_size)
    return 2048


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def _check_modulation_shape(X, scale, shift):
    dim = X.shape[-1]
    assert scale.numel() % dim == 0, "Scale element count must be a multiple of the hidden size."
    n_rows = X.numel() // dim
    scale_rows = scale.numel() // dim
    assert scale_rows > 0, "Scale must have at least one row."
    assert n_rows % scale_rows == 0, "Scale rows must divide hidden state rows for broadcasting."

    if shift is not None:
        assert shift.numel() == scale_rows * dim, "Shift must use the same broadcast rows as scale."

    return scale_rows, n_rows // scale_rows


def _compute_weight_grad(dY, X, RSTD, scale, casting_mode, rows_per_modulation, W):
    if not isinstance(casting_mode, int):
        casting_mode = _str_to_casting_mode[casting_mode]

    dim = X.shape[-1]
    x = X.reshape(-1, dim)
    dy = dY.reshape(-1, dim)
    rstd = RSTD.reshape(-1, 1)
    scale_rows = scale.numel() // dim
    scale_v = scale.reshape(scale_rows, dim)
    mod_idx = torch.arange(x.shape[0], device=x.device, dtype=torch.long) // rows_per_modulation
    scale_b = scale_v[mod_idx]

    x_norm = x.float() * rstd
    mod = 1 + scale_b
    d_rms = dy * mod

    if casting_mode == _CASTING_MODE_LLAMA.value:
        dw = (d_rms * x_norm.to(x.dtype)).float().sum(dim=0)
    elif casting_mode == _CASTING_MODE_GEMMA.value:
        dw = (d_rms.float() * x_norm).sum(dim=0)
    else:
        dw = (d_rms * x_norm).float().sum(dim=0)

    return dw.to(W.dtype)


def modulated_rms_norm_forward(X, W, scale, shift, eps, offset, casting_mode):
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"

    shape = X.shape
    dim = shape[-1]
    scale_rows, rows_per_modulation = _check_modulation_shape(X, scale, shift)

    X = X.view(-1, dim)
    scale = scale.view(scale_rows, dim)
    n_rows, n_cols = X.shape
    X_DTYPE = torch_dtype_to_triton(X.dtype)

    BLOCK_SIZE = get_optimal_block_size(n_cols, True)
    BLOCK_SIZE_M = 2048 // BLOCK_SIZE

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    if W is not None:
        assert X.shape[1] == W.shape[0], "Incompatible hidden size dimension between X and W."
        elementwise_affine = True
    else:
        elementwise_affine = False

    has_shift = shift is not None
    if has_shift:
        shift = shift.view(scale_rows, dim)
    else:
        shift = scale

    num_cores = get_npu_core_count()
    grid_size = min(num_cores * 2, n_rows)

    if n_cols <= 2048:
        _modulated_rms_norm_forward_kernel_no_tiling[(grid_size,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            scale,
            scale.stride(0),
            shift,
            shift.stride(0) if has_shift else 0,
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            offset,
            casting_mode,
            elementwise_affine=elementwise_affine,
            has_shift=has_shift,
            rows_per_modulation=rows_per_modulation,
            X_DTYPE=X_DTYPE,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
    else:
        _modulated_rms_norm_forward_kernel_tiled[(grid_size,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            scale,
            scale.stride(0),
            shift,
            shift.stride(0) if has_shift else 0,
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            offset,
            casting_mode,
            elementwise_affine=elementwise_affine,
            has_shift=has_shift,
            rows_per_modulation=rows_per_modulation,
            X_DTYPE=X_DTYPE,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return Y.view(*shape), RSTD, casting_mode, rows_per_modulation


def modulated_rms_norm_backward(
    dY,
    X,
    W,
    scale,
    shift,
    RSTD,
    offset,
    casting_mode,
    rows_per_modulation,
    in_place,
):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    X = X.view(-1, dim)
    scale_shape = scale.shape
    scale = scale.view(-1, dim)
    n_rows, n_cols = dY.shape
    scale_rows = scale.shape[0]

    num_cores = get_npu_core_count()
    grid_size = min(num_cores * 2, n_rows)

    BLOCK_SIZE = get_optimal_block_size(n_cols, False)
    BLOCK_SIZE_M = 2048 // BLOCK_SIZE

    elementwise_affine = W is not None

    if rows_per_modulation > 1:
        # Per-row dScale/dShift are reduced in PyTorch to preserve bf16 accumulation order.
        dScale = torch.empty((n_rows, n_cols), dtype=X.dtype, device=scale.device)
    else:
        dScale = torch.empty((scale_rows, n_cols), dtype=scale.dtype, device=scale.device)

    has_shift = shift is not None
    if has_shift:
        shift_shape = shift.shape
        if rows_per_modulation > 1:
            dShift = torch.empty((n_rows, n_cols), dtype=shift.dtype, device=shift.device)
        else:
            dShift = torch.empty((scale_rows, n_cols), dtype=shift.dtype, device=shift.device)
    else:
        shift_shape = None
        dShift = dScale

    if in_place:
        dX = dY
    else:
        dX = torch.empty_like(dY)

    use_tiled_backward = n_cols > 2048

    if not use_tiled_backward:
        _modulated_rms_norm_backward_kernel_no_tiling[(grid_size,)](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            scale,
            scale.stride(0),
            RSTD,
            RSTD.stride(0),
            dScale,
            dScale.stride(0),
            dShift,
            dShift.stride(0) if has_shift else 0,
            n_rows,
            n_cols,
            offset,
            casting_mode,
            elementwise_affine=elementwise_affine,
            has_shift=has_shift,
            rows_per_modulation=rows_per_modulation,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
    else:
        _modulated_rms_norm_backward_kernel_tiled[(grid_size,)](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            scale,
            scale.stride(0),
            RSTD,
            RSTD.stride(0),
            dScale,
            dScale.stride(0),
            dShift,
            dShift.stride(0) if has_shift else 0,
            n_rows,
            n_cols,
            offset,
            casting_mode,
            elementwise_affine=elementwise_affine,
            has_shift=has_shift,
            rows_per_modulation=rows_per_modulation,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    dX = dX.view(*shape)
    dW = (
        _compute_weight_grad(dY, X, RSTD, scale, casting_mode, rows_per_modulation, W)
        if elementwise_affine
        else None
    )
    if rows_per_modulation > 1:
        dScale = dScale.view(scale_rows, rows_per_modulation, n_cols).sum(dim=1)
    dScale = dScale.to(scale.dtype).view(*scale_shape) if dScale.dtype != scale.dtype else dScale.view(*scale_shape)
    if has_shift:
        if rows_per_modulation > 1:
            dShift = dShift.view(scale_rows, rows_per_modulation, n_cols).sum(dim=1)
        dShift = dShift.to(shift.dtype).view(*shift_shape) if dShift.dtype != shift.dtype else dShift.view(*shift_shape)
    else:
        dShift = None

    return dX, dW, dScale, dShift


class LigerModulatedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, scale, shift, eps, offset=0.0, casting_mode="llama", in_place=True):
        if isinstance(X, torch.distributed.tensor.DTensor):
            X = X.full_tensor()

        Y, RSTD, casting_mode, rows_per_modulation = modulated_rms_norm_forward(
            X,
            W,
            scale,
            shift,
            eps,
            offset,
            casting_mode,
        )
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.rows_per_modulation = rows_per_modulation
        ctx.has_weight = W is not None
        ctx.has_shift = shift is not None
        if W is not None and shift is not None:
            ctx.save_for_backward(X, W, scale, shift, RSTD)
        elif W is not None:
            ctx.save_for_backward(X, W, scale, RSTD)
        elif shift is not None:
            ctx.save_for_backward(X, scale, shift, RSTD)
        else:
            ctx.save_for_backward(X, scale, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        if isinstance(dY, torch.distributed.tensor.DTensor):
            dY = dY.full_tensor()
        if ctx.has_weight and ctx.has_shift:
            X, W, scale, shift, RSTD = ctx.saved_tensors
        elif ctx.has_weight:
            X, W, scale, RSTD = ctx.saved_tensors
            shift = None
        elif ctx.has_shift:
            X, scale, shift, RSTD = ctx.saved_tensors
            W = None
        else:
            X, scale, RSTD = ctx.saved_tensors
            W = None
            shift = None

        dX, dW, dScale, dShift = modulated_rms_norm_backward(
            dY,
            X,
            W,
            scale,
            shift,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.rows_per_modulation,
            ctx.in_place,
        )

        return dX, dW, dScale, dShift, None, None, None, None
