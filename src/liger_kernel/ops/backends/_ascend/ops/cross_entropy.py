from typing import Optional

import torch
import triton
import triton.language as tl

from triton.language.math import tanh

from liger_kernel.ops.backends._ascend.ub_manager import compute_default_tiling_strategy
from liger_kernel.ops.utils import get_npu_core_count


@triton.jit
def liger_cross_entropy_forward_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    lse_ptr,
    token_accuracy_ptr,
    token_accuracy_stride,
    predicted_tokens_ptr,
    predicted_tokens_stride,
    n_cols,
    n_rows,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    RETURN_Z_LOSS: tl.constexpr,
    SAVE_LSE: tl.constexpr,
    RETURN_TOKEN_ACCURACY: tl.constexpr,
    RETURN_PREDICTED_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
):
    """
    This kernel computes cross entropy forward outputs only.
    """

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    start_row = pid
    stride = num_progs

    for row_idx in range(start_row, n_rows, stride):
        program_id = row_idx.to(tl.int64)
        y = tl.load(Y_ptr + program_id)
        X_ptr_offset = program_id * X_stride

        if y == ignore_index:
            if SAVE_LSE:
                tl.store(lse_ptr + program_id, 0.0)
            if RETURN_TOKEN_ACCURACY:
                token_accuracy_ptr_offset = program_id * token_accuracy_stride
                tl.store(token_accuracy_ptr + token_accuracy_ptr_offset, 0.0)
            if RETURN_PREDICTED_TOKENS:
                predicted_tokens_ptr_offset = program_id * predicted_tokens_stride
                tl.store(predicted_tokens_ptr + predicted_tokens_ptr_offset, -1)
        else:
            if RETURN_Z_LOSS:
                z_loss_ptr_offset = program_id
            if RETURN_TOKEN_ACCURACY:
                token_accuracy_ptr_offset = program_id * token_accuracy_stride
            if RETURN_PREDICTED_TOKENS:
                predicted_tokens_ptr_offset = program_id * predicted_tokens_stride

            if HAS_WEIGHT:
                weight_y = tl.load(weight_ptr + y).cast(tl.float32)

            m = float("-inf")
            d = 0.0
            argmax_idx = 0
            ori_X_y = tl.load(X_ptr + X_ptr_offset + y).cast(tl.float32)
            if HAS_SOFTCAPPING:
                ori_X_y = softcap * tanh(ori_X_y / softcap)

            scaled_x_sum = 0.0
            eps = label_smoothing / n_cols

            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_block = tl.load(
                    X_ptr + X_ptr_offset + X_offsets,
                    mask=X_offsets < n_cols,
                    other=float("-inf"),
                ).cast(tl.float32)
                if HAS_SOFTCAPPING:
                    X_block = softcap * tanh(X_block / softcap)
                block_max = tl.max(X_block)

                if RETURN_TOKEN_ACCURACY or RETURN_PREDICTED_TOKENS:
                    is_max_mask = X_block == block_max
                    masked_offsets = X_offsets + (n_cols - X_offsets) * (1 - is_max_mask.to(tl.int64))
                    current_block_argmax_idx = tl.min(masked_offsets)
                    is_new_max = block_max > m
                    argmax_idx = argmax_idx + is_new_max.to(tl.int64) * (current_block_argmax_idx - argmax_idx)

                if label_smoothing > 0:
                    valid_mask = (X_offsets < n_cols).to(tl.float32)
                    if HAS_WEIGHT:
                        weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                        scaled_x_sum += tl.sum(-eps * X_block * weight_block * valid_mask)
                    else:
                        scaled_x_sum += tl.sum(-eps * X_block * valid_mask)

                m_new = tl.maximum(m, block_max)
                d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
                m = m_new

            lse = m + tl.log(d)
            if SAVE_LSE:
                tl.store(lse_ptr + program_id, lse)
            loss = lse - ori_X_y
            if HAS_WEIGHT:
                loss = weight_y * loss

            if label_smoothing > 0:
                if HAS_WEIGHT:
                    smooth_loss = scaled_x_sum + eps * lse * weight_sum
                else:
                    smooth_loss = scaled_x_sum + label_smoothing * lse
                loss = loss * (1 - label_smoothing) + smooth_loss

            z_loss = lse_square_scale * lse * lse
            if reduction == "mean":
                if HAS_WEIGHT:
                    loss = loss / sum_non_ignore_weight
                else:
                    loss = loss / n_non_ignore
                z_loss = z_loss / n_non_ignore
            loss += z_loss

            tl.store(loss_ptr + program_id, loss)
            if RETURN_Z_LOSS:
                tl.store(z_loss_ptr + program_id, z_loss)
            if RETURN_TOKEN_ACCURACY:
                is_correct = 1.0 if argmax_idx == y else 0.0
                tl.store(token_accuracy_ptr + token_accuracy_ptr_offset, is_correct)
            if RETURN_PREDICTED_TOKENS:
                tl.store(predicted_tokens_ptr + predicted_tokens_ptr_offset, argmax_idx)


@triton.jit
def liger_cross_entropy_backward_kernel_no_weight(
    X_ptr,
    X_stride,
    Y_ptr,
    lse_ptr,
    grad_output_ptr,
    grad_output_stride,
    dX_ptr,
    dX_stride,
    n_cols,
    n_rows,
    n_non_ignore,
    ignore_index,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRAD_OUTPUT_VECTOR: tl.constexpr,
):
    """
    Specialized backward kernel for the common path without class weights, softcap, z-loss, or label smoothing.
    Optimized for Ascend NPU memory bandwidth utilization.
    """

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    start_row = pid
    stride = num_progs
    
    scale_factor = 1.0
    if reduction == "mean":
        scale_factor = 1.0 / n_non_ignore

    for row_idx in range(start_row, n_rows, stride):
        program_id = row_idx.to(tl.int64)
        y = tl.load(Y_ptr + program_id)
        X_ptr_offset = program_id * X_stride
        dX_ptr_offset = program_id * dX_stride
        
        grad_scale = (
            tl.load(grad_output_ptr + program_id * grad_output_stride)
            if HAS_GRAD_OUTPUT_VECTOR
            else tl.load(grad_output_ptr)
        ).cast(tl.float32)
        
        final_scale = grad_scale * scale_factor

        if y == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + dX_ptr_offset + X_offsets, 0.0, mask=X_offsets < n_cols)
        else:
            lse = tl.load(lse_ptr + program_id).cast(tl.float32)

            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_block = tl.load(
                    X_ptr + X_ptr_offset + X_offsets,
                    mask=X_offsets < n_cols,
                    other=float("-inf"),
                ).cast(tl.float32)
                
                softmax = tl.exp(X_block - lse)
                
                y_in_block = (y >= i) & (y < i + BLOCK_SIZE)
                if y_in_block:
                    mask = (X_offsets == y).to(tl.float32)
                    softmax = softmax - mask
                
                grad = softmax * final_scale
                tl.store(dX_ptr + dX_ptr_offset + X_offsets, grad, mask=X_offsets < n_cols)


@triton.jit
def liger_cross_entropy_backward_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    weight_ptr,
    lse_ptr,
    grad_output_ptr,
    grad_output_stride,
    dX_ptr,
    dX_stride,
    n_cols,
    n_rows,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    HAS_GRAD_OUTPUT_VECTOR: tl.constexpr,
):
    """
    This kernel computes cross entropy gradients only.
    """

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    start_row = pid
    stride = num_progs
    inv_n_non_ignore = 1.0
    inv_sum_non_ignore_weight = 1.0
    if reduction == "mean":
        inv_n_non_ignore = 1.0 / n_non_ignore
        if HAS_WEIGHT:
            inv_sum_non_ignore_weight = 1.0 / sum_non_ignore_weight

    for row_idx in range(start_row, n_rows, stride):
        program_id = row_idx.to(tl.int64)
        y = tl.load(Y_ptr + program_id)
        X_ptr_offset = program_id * X_stride
        dX_ptr_offset = program_id * dX_stride
        grad_scale = (
            tl.load(grad_output_ptr + program_id * grad_output_stride)
            if HAS_GRAD_OUTPUT_VECTOR
            else tl.load(grad_output_ptr)
        ).cast(tl.float32)

        if y == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + dX_ptr_offset + X_offsets, 0.0, mask=X_offsets < n_cols)
        else:
            if HAS_WEIGHT:
                weight_y = tl.load(weight_ptr + y).cast(tl.float32)
            lse = tl.load(lse_ptr + program_id).cast(tl.float32)
            eps = label_smoothing / n_cols
            eps_weight_sum = eps * weight_sum
            z_scale = 1.0 + 2.0 * lse_square_scale * lse
            one_minus_ls = 1.0 - label_smoothing
            z_deriv = 2.0 * lse_square_scale * lse

            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_block = tl.load(
                    X_ptr + X_ptr_offset + X_offsets,
                    mask=X_offsets < n_cols,
                    other=float("-inf"),
                ).cast(tl.float32)
                if HAS_SOFTCAPPING:
                    intermediate = tanh(X_block / softcap)
                    X_block = softcap * intermediate

                softmax_X = tl.exp(X_block - lse)
                if not HAS_WEIGHT:
                    X_block = softmax_X * z_scale - eps
                    if y >= i and y < i + BLOCK_SIZE:
                        y_mask = (X_offsets == y).to(tl.float32)
                        X_block = X_block - y_mask * one_minus_ls
                    if reduction == "mean":
                        X_block = X_block * inv_n_non_ignore
                else:
                    weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                    dloss_ori = one_minus_ls * softmax_X
                    if y >= i and y < i + BLOCK_SIZE:
                        y_mask = (X_offsets == y).to(tl.float32)
                        dloss_ori = dloss_ori - y_mask * one_minus_ls
                    dloss_ori = dloss_ori * weight_y
                    dloss_smooth = -eps * weight_block + softmax_X * eps_weight_sum
                    dz_loss = z_deriv * softmax_X
                    if reduction == "mean":
                        dloss_ori = dloss_ori * inv_sum_non_ignore_weight
                        dloss_smooth = dloss_smooth * inv_sum_non_ignore_weight
                        dz_loss = dz_loss * inv_n_non_ignore
                    X_block = dloss_ori + dloss_smooth + dz_loss

                if HAS_SOFTCAPPING:
                    X_block = X_block * (1 - intermediate * intermediate)

                X_block = X_block * grad_scale
                tl.store(dX_ptr + dX_ptr_offset + X_offsets, X_block, mask=X_offsets < n_cols)


def get_optimal_block_size(n_cols, has_gradients=True):
    """
    Calculate optimal Block Size using compute_default_tiling_strategy
    Optimized for Ascend NPU memory bandwidth utilization.
    """
    if has_gradients:
        if n_cols <= 8192:
            return 512
        if n_cols <= 32768:
            return 1024
        if n_cols <= 131072:
            return 2048
        return 4096

    multiplier = 12.0 if has_gradients else 8.0

    tile_shapes = compute_default_tiling_strategy(
        safety_margin=0.9, dtype_size=4, memory_multiplier=multiplier, shapes=((n_cols,),), tiling_dims=(0,)
    )

    if tile_shapes and len(tile_shapes) > 0:
        block_size = tile_shapes[0][0]
        return block_size
    else:
        return 4096


def _compute_cross_entropy_stats(target, ignore_index, weight):
    target_mask = target != ignore_index
    n_non_ignore = target_mask.sum().item()
    sum_non_ignore_weight = n_non_ignore
    weight_sum = 0.0
    if weight is not None:
        non_ignore_targets = target.masked_select(target_mask)
        sum_non_ignore_weight = torch.gather(weight, dim=0, index=non_ignore_targets).sum().item()
        weight_sum = weight.sum().item()
    return n_non_ignore, sum_non_ignore_weight, weight_sum


def cross_entropy_forward(
    _input,
    target,
    weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
    save_lse=False,
    return_token_accuracy=False,
    return_predicted_tokens=False,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    assert isinstance(return_token_accuracy, bool), (
        f"return_token_accuracy must be True or False. Got: {return_token_accuracy}"
    )
    assert isinstance(return_predicted_tokens, bool), (
        f"return_predicted_tokens must be True or False. Got: {return_predicted_tokens}"
    )

    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = get_optimal_block_size(V, has_gradients=False)

    # unreduced loss
    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)
    z_loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device) if return_z_loss else None
    lse_1d = torch.empty(n_rows, dtype=torch.float32, device=_input.device) if save_lse else loss_1d
    token_accuracy_1d = (
        torch.zeros(n_rows, dtype=torch.float32, device=_input.device) if return_token_accuracy else None
    )
    predicted_tokens_1d = (
        torch.full((n_rows,), -1, dtype=torch.int64, device=_input.device) if return_predicted_tokens else None
    )

    target_mask = target != ignore_index
    n_non_ignore, sum_non_ignore_weight, weight_sum = _compute_cross_entropy_stats(target, ignore_index, weight)
    assert (target * target_mask).max() < _input.shape[-1], (
        f"Target {target.max()} is out of bounds. Expected < {_input.shape[-1]}"
    )
    assert (target * target_mask).min() >= 0, f"Target {target.min()} is out of bounds. Expected >= 0"
    if weight is not None:
        assert weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {weight.shape}"
        assert torch.is_floating_point(weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {weight.dtype}"
        )
        # ensure weight is contiguous
        if weight.stride(-1) != 1:
            weight = weight.contiguous()

    # ensure _input and target are contiguous in the last dimension
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    # NPU-optimized grid configuration
    grid_size = min(get_npu_core_count(), n_rows)

    liger_cross_entropy_forward_kernel[(grid_size,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        weight_ptr=weight,
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_1d,
        lse_ptr=lse_1d,
        token_accuracy_ptr=token_accuracy_1d,
        token_accuracy_stride=token_accuracy_1d.stride(-1)
        if return_token_accuracy
        else 0,  # always 1 if accuracy is enabled
        predicted_tokens_ptr=predicted_tokens_1d,
        predicted_tokens_stride=predicted_tokens_1d.stride(-1)
        if return_predicted_tokens
        else 0,  # always 1 if predicted tokens is enabled
        n_cols=V,
        n_rows=n_rows,
        n_non_ignore=n_non_ignore,
        sum_non_ignore_weight=sum_non_ignore_weight,
        ignore_index=ignore_index,
        weight_sum=weight_sum,
        lse_square_scale=lse_square_scale,
        label_smoothing=label_smoothing,
        reduction=reduction,
        softcap=softcap,
        RETURN_Z_LOSS=return_z_loss,
        SAVE_LSE=save_lse,
        RETURN_TOKEN_ACCURACY=return_token_accuracy,
        RETURN_PREDICTED_TOKENS=return_predicted_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=True if weight is not None else False,
        HAS_SOFTCAPPING=True if softcap is not None else False,
    )

    if reduction == "none":
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None
        token_accuracy = token_accuracy_1d if return_token_accuracy else None
    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
        # For accuracy, we compute the mean across all non-ignored tokens
        token_accuracy = torch.sum(token_accuracy_1d) / n_non_ignore if return_token_accuracy else None

    predicted_tokens = predicted_tokens_1d if return_predicted_tokens else None

    return (
        loss,
        z_loss,
        token_accuracy,
        predicted_tokens,
        _input,
        lse_1d if save_lse else None,
        n_non_ignore,
        sum_non_ignore_weight,
        weight_sum,
    )


def cross_entropy_backward(
    _input,
    target,
    weight,
    lse,
    grad_output,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    n_non_ignore=None,
    sum_non_ignore_weight=None,
    weight_sum=None,
):
    BT, V = _input.shape
    n_rows = BT
    BLOCK_SIZE = get_optimal_block_size(V, has_gradients=True)

    if n_non_ignore is None or sum_non_ignore_weight is None or weight_sum is None:
        n_non_ignore, sum_non_ignore_weight, weight_sum = _compute_cross_entropy_stats(target, ignore_index, weight)
    if weight is not None and weight.stride(-1) != 1:
        weight = weight.contiguous()

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()
    if grad_output.ndim > 0 and grad_output.stride(-1) != 1:
        grad_output = grad_output.contiguous()

    grad_input = torch.empty_like(_input)
    grid_size = min(get_npu_core_count(), n_rows)
    grad_output_stride = grad_output.stride(-1) if grad_output.ndim > 0 else 0

    use_no_weight_fast_path = weight is None and softcap is None and label_smoothing == 0.0 and lse_square_scale == 0.0

    if use_no_weight_fast_path:
        liger_cross_entropy_backward_kernel_no_weight[(grid_size,)](
            X_ptr=_input,
            X_stride=_input.stride(-2),
            Y_ptr=target,
            lse_ptr=lse,
            grad_output_ptr=grad_output,
            grad_output_stride=grad_output_stride,
            dX_ptr=grad_input,
            dX_stride=grad_input.stride(-2),
            n_cols=V,
            n_rows=n_rows,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            reduction=reduction,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_GRAD_OUTPUT_VECTOR=grad_output.ndim > 0,
        )
    else:
        liger_cross_entropy_backward_kernel[(grid_size,)](
            X_ptr=_input,
            X_stride=_input.stride(-2),
            Y_ptr=target,
            weight_ptr=weight,
            lse_ptr=lse,
            grad_output_ptr=grad_output,
            grad_output_stride=grad_output_stride,
            dX_ptr=grad_input,
            dX_stride=grad_input.stride(-2),
            n_cols=V,
            n_rows=n_rows,
            n_non_ignore=n_non_ignore,
            sum_non_ignore_weight=sum_non_ignore_weight,
            weight_sum=weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=True if weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            HAS_GRAD_OUTPUT_VECTOR=grad_output.ndim > 0,
        )

    return grad_input


class LigerCrossEntropyFunction(torch.autograd.Function):
    """
    This class implements a custom autograd function for the Liger Cross Entropy loss.
    It overrides the forward and backward methods of the torch.autograd.Function class.
    """

    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.FloatTensor],
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        return_token_accuracy: bool = False,
        return_predicted_tokens: bool = False,
    ):
        """
        The forward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object.
        _input (tensor): The input tensor of shape (BT, V) where B is batch size, T is sequence length, V is vocab size.
        target (tensor): The target tensor of shape (BT) where each value is in [0, V-1].
        weight(Tensor, optional): a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index (int): The index to ignore in the target.
        lse_square_scale (float): The scaler of (logsumexp(_input)) ^ 2 adding to the loss for the stability of training.
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str): The reduction to apply to the output: "none" | "mean | "sum".
        softcap (Optional[float]): The upper threshold for scaling logits to the range (-softcap, +softcap).
        return_z_loss (bool): When `return_z_loss` is `True`, returns (loss, z_loss, token_accuracy, predicted_tokens) instead of (loss, None, None, None). Default: `False`
        return_token_accuracy (bool): When `return_token_accuracy` is `True`, computes and returns per-token accuracy without materializing logits. Default: `False`
        return_predicted_tokens (bool): When `return_predicted_tokens` is `True`, returns per-token predicted class indices (argmax) without materializing logits. Default: `False`

        Returns:
        tuple: A tuple with the computed losses, accuracy, and predicted tokens: (loss, z_loss, token_accuracy, predicted_tokens). z_loss, token_accuracy, and predicted_tokens are None if not requested.
        """
        input_requires_grad = _input.requires_grad

        (
            loss,
            z_loss,
            token_accuracy,
            predicted_tokens,
            _input,
            lse,
            n_non_ignore,
            sum_non_ignore_weight,
            weight_sum,
        ) = cross_entropy_forward(
            _input,
            target,
            weight,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
            return_z_loss,
            save_lse=input_requires_grad,
            return_token_accuracy=return_token_accuracy,
            return_predicted_tokens=return_predicted_tokens,
        )
        if input_requires_grad:
            saved_tensors = [_input.detach(), target.detach(), lse.detach()]
            if weight is not None:
                saved_tensors.append(weight.detach())
            ctx.save_for_backward(*saved_tensors)
        ctx.n_non_ignore = n_non_ignore
        ctx.sum_non_ignore_weight = sum_non_ignore_weight
        ctx.weight_sum = weight_sum
        ctx.has_weight = weight is not None
        ctx.ignore_index = ignore_index
        ctx.lse_square_scale = lse_square_scale
        ctx.label_smoothing = label_smoothing
        ctx.reduction = reduction
        ctx.softcap = softcap
        ctx.return_z_loss = return_z_loss
        ctx.return_token_accuracy = return_token_accuracy
        ctx.return_predicted_tokens = return_predicted_tokens

        return loss, z_loss, token_accuracy, predicted_tokens

    @staticmethod
    def backward(ctx, grad_output, grad_output2, grad_output3, grad_output4):
        """
        The backward pass of the Liger Cross Entropy loss.

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The tensor containing the gradient of the loss with respect to the output.
        grad_output2 (tensor): No use. Gradient for z_loss (not used as z_loss is only for logging).
        grad_output3 (tensor): No use. Gradient for token_accuracy (not used as token_accuracy is only for metrics).
        grad_output4 (tensor): No use. Gradient for predicted_tokens (not used as predicted_tokens is only for metrics).
        Returns:
        tuple: A tuple with the gradients with respect to the inputs. The elements are tensors or None.
        """
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        if ctx.return_token_accuracy:
            del grad_output3  # token_accuracy is only for metrics
        if ctx.return_predicted_tokens:
            del grad_output4  # predicted_tokens is only for metrics

        if ctx.has_weight:
            _input, target, lse, weight = ctx.saved_tensors
        else:
            _input, target, lse = ctx.saved_tensors
            weight = None
        _input = cross_entropy_backward(
            _input,
            target,
            weight,
            lse,
            grad_output,
            ctx.ignore_index,
            ctx.lse_square_scale,
            ctx.label_smoothing,
            ctx.reduction,
            ctx.softcap,
            ctx.n_non_ignore,
            ctx.sum_non_ignore_weight,
            ctx.weight_sum,
        )
        return (
            _input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
