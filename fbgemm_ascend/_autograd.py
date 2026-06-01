import logging

import torch


def _infer_max_lengths_from_dense(dense: torch.Tensor, offset_count: int, values_dim: int) -> list[int]:
    output_without_inner = values_dim == 1
    start = 1
    end = dense.dim() if output_without_inner else dense.dim() - 1
    max_lengths = list(dense.shape[start:end])
    if len(max_lengths) != offset_count:
        raise RuntimeError(
            f"dense dim mismatch for {offset_count} offsets and values dim {values_dim}: got dense shape {tuple(dense.shape)}"
        )
    return max_lengths


def _jagged_dense_elementwise_add_jagged_output_setup_context(ctx, inputs, output) -> None:
    x_values, x_offsets, y = inputs
    ctx.save_for_backward(x_values, y, *x_offsets)
    ctx.offset_count = len(x_offsets)


def _jagged_dense_elementwise_add_jagged_output_backward(ctx, grad_output, grad_offsets=None):
    saved = ctx.saved_tensors
    x_values = saved[0]
    y = saved[1]
    x_offsets = list(saved[2 : 2 + ctx.offset_count])
    grad_x = grad_output.contiguous()
    if grad_output.numel() == 0 or x_values.numel() == 0:
        grad_y = torch.zeros_like(y)
    else:
        max_lengths = _infer_max_lengths_from_dense(y, len(x_offsets), x_values.dim())
        grad_y = torch.ops.fbgemm.jagged_to_padded_dense(grad_x, x_offsets, max_lengths, 0.0)
    return grad_x, [None for _ in x_offsets], grad_y


def _jagged_dense_elementwise_mul_setup_context(ctx, inputs, output) -> None:
    x_values, x_offsets, y = inputs
    ctx.save_for_backward(x_values, y, *x_offsets)
    ctx.offset_count = len(x_offsets)


def _jagged_dense_elementwise_mul_backward(ctx, grad_output, grad_offsets=None):
    saved = ctx.saved_tensors
    x_values = saved[0]
    y = saved[1]
    x_offsets = list(saved[2 : 2 + ctx.offset_count])
    if grad_output.numel() == 0 or x_values.numel() == 0:
        return torch.zeros_like(x_values), [None for _ in x_offsets], torch.zeros_like(y)
    grad_x, grad_y = torch.ops.fbgemm.jagged_dense_elementwise_mul_backward(
        grad_output.contiguous(),
        x_offsets,
        y,
        x_values,
    )
    return grad_x, [None for _ in x_offsets], grad_y


def _jagged_1d_to_dense_setup_context(ctx, inputs, output) -> None:
    values, offsets = inputs[0], inputs[1]
    ctx.save_for_backward(offsets)
    ctx.total_L = values.size(0)


def _jagged_1d_to_dense_backward(ctx, grad_output):
    offsets = ctx.saved_tensors[0]
    grad_input = torch.ops.fbgemm.jagged_to_padded_dense_backward(
        grad_output,
        [offsets],
        ctx.total_L,
    )
    return grad_input, None, None, None


def register_python_autograd() -> None:
    try:
        torch.library.register_autograd(
            "fbgemm::jagged_dense_elementwise_add_jagged_output",
            _jagged_dense_elementwise_add_jagged_output_backward,
            setup_context=_jagged_dense_elementwise_add_jagged_output_setup_context,
        )
        torch.library.register_autograd(
            "fbgemm::jagged_dense_elementwise_mul",
            _jagged_dense_elementwise_mul_backward,
            setup_context=_jagged_dense_elementwise_mul_setup_context,
        )
        torch.library.register_autograd(
            "fbgemm::jagged_1d_to_dense",
            _jagged_1d_to_dense_backward,
            setup_context=_jagged_1d_to_dense_setup_context,
        )
    except RuntimeError as e:
        logging.warning("fbgemm_ascend: Python autograd registration skipped: %s", e)
