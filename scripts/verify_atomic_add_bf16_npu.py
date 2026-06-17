#!/usr/bin/env python3
"""Verify triton-ascend tl.atomic_add precision for bf16 on NPU.

Compares bf16 atomic accumulation against fp32 reference reductions and an
fp32 atomic_add workaround. Run on Ascend NPU:

    python scripts/verify_atomic_add_bf16_npu.py
"""

from __future__ import annotations

import argparse
import sys

import torch
import triton
import triton.language as tl

from liger_kernel.utils import is_npu_available


def get_device() -> torch.device:
    if is_npu_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        print("WARNING: NPU not available, falling back to CUDA for smoke test.")
        return torch.device("cuda")
    raise RuntimeError("Neither NPU nor CUDA is available.")


@triton.jit
def _atomic_add_column_kernel_bf16(
    out_ptr,
    in_ptr,
    in_row_stride,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    """Each program (row) atomically adds one row into a shared output vector."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < n_cols
    row_ptr = in_ptr + row_idx * in_row_stride
    vals = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + col_offsets, vals, sem="relaxed", mask=mask)


@triton.jit
def _atomic_add_column_kernel_fp32(
    out_ptr,
    in_ptr,
    in_row_stride,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < n_cols
    row_ptr = in_ptr + row_idx * in_row_stride
    vals = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(out_ptr + col_offsets, vals, sem="relaxed", mask=mask)


@triton.jit
def _atomic_add_scalar_kernel_bf16(
    out_ptr,
    in_ptr,
    n_elems,
    BLOCK: tl.constexpr,
):
    """Many programs atomically add into out_ptr[0] (stress many-to-one contention)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(vals, axis=0)
    tl.atomic_add(out_ptr, partial, sem="relaxed")


@triton.jit
def _atomic_add_scalar_kernel_fp32(
    out_ptr,
    in_ptr,
    n_elems,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elems
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(vals, axis=0)
    tl.atomic_add(out_ptr, partial, sem="relaxed")


def _launch_column_atomic(
    inp: torch.Tensor,
    dtype: torch.dtype,
    kernel,
) -> torch.Tensor:
    n_rows, n_cols = inp.shape
    block_n = triton.next_power_of_2(n_cols)
    out = torch.zeros(n_cols, dtype=dtype, device=inp.device)
    grid = (n_rows,)
    kernel[grid](
        out,
        inp,
        inp.stride(0),
        n_cols,
        BLOCK_N=block_n,
    )
    return out


def _launch_scalar_atomic(
    values: torch.Tensor,
    dtype: torch.dtype,
    kernel,
    block: int = 256,
) -> torch.Tensor:
    n_elems = values.numel()
    n_progs = triton.cdiv(n_elems, block)
    out = torch.zeros(1, dtype=dtype, device=values.device)
    grid = (n_progs,)
    kernel[grid](out, values, n_elems, BLOCK=block)
    return out


def _report(name: str, got: torch.Tensor, expected: torch.Tensor) -> dict:
    got_f = got.float().cpu()
    exp_f = expected.float().cpu()
    abs_err = (got_f - exp_f).abs()
    rel_err = abs_err / exp_f.abs().clamp_min(1e-12)

    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    max_rel = rel_err.max().item()

    # Classify by worst absolute error only (ignore blow-ups near zero in rel_err).
    severe = max_abs > 0.5
    mild = max_abs <= 0.0625
    if severe:
        status = "FAIL (severe precision loss)"
    elif mild:
        status = "OK (within bf16 rounding)"
    else:
        status = "WARN (moderate error)"

    print(f"\n=== {name} ===")
    print(f"  shape: {tuple(got.shape)}, dtype: {got.dtype}")
    print(f"  max abs err: {max_abs:.6g}, mean abs err: {mean_abs:.6g}, max rel err: {max_rel:.6g}")
    print(f"  status: {status}")

    worst_idx = abs_err.argmax().item()
    if got.numel() <= 8:
        print(f"  got:      {got_f.flatten().tolist()}")
        print(f"  expected: {exp_f.flatten().tolist()}")
    else:
        print(f"  worst idx {worst_idx}: got={got_f.flatten()[worst_idx]:.6g}, expected={exp_f.flatten()[worst_idx]:.6g}")

    return {
        "name": name,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "status": status,
        "severe": severe,
    }


def run_column_reduction(device: torch.device, n_rows: int, n_cols: int, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    inp = torch.randn(n_rows, n_cols, device=device, dtype=torch.bfloat16)

    # Reference: accumulate in fp32, cast once (ideal reduction).
    expected_bf16 = inp.float().sum(dim=0).to(torch.bfloat16)
    expected_fp32 = inp.float().sum(dim=0)

    got_bf16 = _launch_column_atomic(inp, torch.bfloat16, _atomic_add_column_kernel_bf16)
    got_fp32 = _launch_column_atomic(inp, torch.float32, _atomic_add_column_kernel_fp32).to(torch.bfloat16)

    return [
        _report(f"column reduction bf16 atomic_add ({n_rows}x{n_cols})", got_bf16, expected_bf16),
        _report(f"column reduction fp32 atomic_add workaround ({n_rows}x{n_cols})", got_fp32, expected_bf16),
        _report(
            f"bf16 vs fp32 atomic_add delta ({n_rows}x{n_cols})",
            got_bf16,
            got_fp32,
        ),
    ]


def run_many_small_adds(device: torch.device, n_rows: int, n_cols: int) -> list[dict]:
    # Each row contributes 1/256 in bf16; fp32 sum should be n_rows/256.
    small = torch.tensor(1.0 / 256.0, device=device, dtype=torch.bfloat16)
    inp = small.expand(n_rows, n_cols).contiguous()

    expected_bf16 = inp.float().sum(dim=0).to(torch.bfloat16)
    got_bf16 = _launch_column_atomic(inp, torch.bfloat16, _atomic_add_column_kernel_bf16)
    got_fp32 = _launch_column_atomic(inp, torch.float32, _atomic_add_column_kernel_fp32).to(torch.bfloat16)

    return [
        _report(f"many small bf16 adds ({n_rows}x{n_cols}, val=1/256)", got_bf16, expected_bf16),
        _report(f"many small fp32 atomic workaround ({n_rows}x{n_cols})", got_fp32, expected_bf16),
    ]


def run_scalar_contention(device: torch.device, n_elems: int, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    values = torch.randn(n_elems, device=device, dtype=torch.bfloat16)

    expected_bf16 = values.float().sum().to(torch.bfloat16)
    got_bf16 = _launch_scalar_atomic(values, torch.bfloat16, _atomic_add_scalar_kernel_bf16)[0]
    got_fp32 = _launch_scalar_atomic(values, torch.float32, _atomic_add_scalar_kernel_fp32)[0].to(torch.bfloat16)

    return [
        _report(f"scalar contention bf16 atomic_add (n={n_elems})", got_bf16, expected_bf16),
        _report(f"scalar contention fp32 atomic_add workaround (n={n_elems})", got_fp32, expected_bf16),
        _report("scalar bf16 vs fp32 atomic_add delta", got_bf16, got_fp32),
    ]


def run_modulated_rms_norm_pattern(device: torch.device, n_rows: int, n_cols: int, rows_per_mod: int, seed: int) -> list[dict]:
    """Mimic dshift accumulation: rows_per_modulation rows share one shift gradient row."""
    torch.manual_seed(seed)
    dy = torch.randn(n_rows, n_cols, device=device, dtype=torch.bfloat16)

    # Reference: group rows and sum dY within each modulation group, then broadcast rows share cols.
    n_mod_rows = n_rows // rows_per_mod
    expected = torch.zeros(n_mod_rows, n_cols, device=device, dtype=torch.bfloat16)
    for r in range(n_rows):
        mod_row = r // rows_per_mod
        expected[mod_row] += dy[r].float()
    expected = expected.to(torch.bfloat16)

    # Kernel view: each X row atomically adds into its modulation row (like rows_per_modulation > 1).
    block_n = triton.next_power_of_2(n_cols)
    out = torch.zeros(n_mod_rows, n_cols, device=device, dtype=torch.bfloat16)

    @triton.jit
    def _dshift_pattern_kernel(dshift_ptr, dy_ptr, dy_row_stride, dshift_row_stride, n_cols, rows_per_mod, BLOCK_N: tl.constexpr):
        row_idx = tl.program_id(0)
        mod_row_idx = row_idx // rows_per_mod
        col_offsets = tl.arange(0, BLOCK_N)
        mask = col_offsets < n_cols
        dy_row_ptr = dy_ptr + row_idx * dy_row_stride
        dshift_row_ptr = dshift_ptr + mod_row_idx * dshift_row_stride
        dy_vals = tl.load(dy_row_ptr + col_offsets, mask=mask, other=0.0)
        tl.atomic_add(dshift_row_ptr + col_offsets, dy_vals, sem="relaxed", mask=mask)

    _dshift_pattern_kernel[(n_rows,)](
        out,
        dy,
        dy.stride(0),
        out.stride(0),
        n_cols,
        rows_per_mod,
        BLOCK_N=block_n,
    )

    return [_report(f"dshift pattern bf16 atomic_add ({n_rows}x{n_cols}, rpm={rows_per_mod})", out, expected)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rows", type=int, default=4096)
    parser.add_argument("--n-cols", type=int, default=256)
    parser.add_argument("--scalar-n", type=int, default=65536)
    parser.add_argument("--rows-per-mod", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Triton: {triton.__version__}")

    results: list[dict] = []
    results.extend(run_column_reduction(device, args.n_rows, args.n_cols, args.seed))
    results.extend(run_many_small_adds(device, args.n_rows, args.n_cols))
    results.extend(run_scalar_contention(device, args.scalar_n, args.seed))
    results.extend(
        run_modulated_rms_norm_pattern(device, args.n_rows, args.n_cols, args.rows_per_mod, args.seed)
    )

    print("\n========== Summary ==========")
    bf16_cases = [r for r in results if "bf16" in r["name"] and "delta" not in r["name"] and "fp32" not in r["name"]]
    fp32_ref_cases = [r for r in results if "fp32 atomic" in r["name"] and "workaround" in r["name"]]
    bf16_severe = [r for r in bf16_cases if r["severe"]]
    fp32_severe = [r for r in fp32_ref_cases if r["severe"]]

    if bf16_severe:
        print(f"bf16 atomic_add: {len(bf16_severe)}/{len(bf16_cases)} case(s) with severe error.")
        for r in bf16_severe:
            print(f"  - {r['name']}: max_abs={r['max_abs']:.6g}")
    else:
        print("bf16 atomic_add: no severe errors in tested cases.")

    if fp32_severe:
        print(f"fp32 atomic_add workaround: {len(fp32_severe)} unexpected severe failure(s).")
    else:
        print("fp32 atomic_add workaround: matches fp32-sum reference.")

    if bf16_severe and not fp32_severe:
        print(
            "\nConclusion: bf16 tl.atomic_add on NPU loses precision under contention. "
            "Use fp32 buffers for accumulation, then cast to bf16 once."
        )
        return 1
    if bf16_severe:
        print("\nConclusion: both bf16 and fp32 atomic_add diverge — investigate further.")
        return 2
    print("\nConclusion: no severe bf16 atomic_add errors detected in these cases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
