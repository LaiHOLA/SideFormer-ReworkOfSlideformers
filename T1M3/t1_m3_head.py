
# -*- coding: utf-8 -*-
"""
t1_m3_head.py

Exact block-wise (chunked) cross-entropy without materializing full logits.
Goal: reduce peak GPU memory at the output layer (B*S*V logits).

Forward:
  - stream over vocab blocks, compute online logsumexp (stable) + gather true logits
Backward:
  - recompute block logits, compute block softmax, accumulate grad_hidden (and optionally grad_weight/bias)

This file is self-contained (no CUDA extension compile). It uses GEMMs on GPU via PyTorch.
Works with bf16/fp16 hidden/weight; uses fp32 accumulation for logsumexp/softmax.

Usage:
  from t1_m3_head import chunked_ce_loss

  loss = chunked_ce_loss(hidden_2d, weight, target, chunk_size=8192, reduction="mean")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ChunkedCEConfig:
    chunk_size: int = 8192          # vocab block size
    reduction: str = "mean"         # "mean" | "sum" | "none"
    # numerical stability & perf
    compute_dtype: torch.dtype = torch.float32  # used for logsumexp/softmax math
    allow_tf32: bool = True         # for matmul on Ampere+


class _ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_2d: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
                bias: Optional[torch.Tensor], chunk_size: int, reduction: str,
                compute_dtype: torch.dtype, allow_tf32: bool):
        """
        hidden_2d: [N, H] on CUDA
        weight:    [V, H] on CUDA (or at least same device as hidden_2d)
        target:    [N] long, values in [0, V)
        bias:      [V] optional
        """
        assert hidden_2d.is_cuda, "hidden_2d must be CUDA tensor"
        assert weight.is_cuda, "weight must be CUDA tensor in this implementation"
        assert target.dtype == torch.long and target.dim() == 1
        assert hidden_2d.dim() == 2 and weight.dim() == 2
        N, H = hidden_2d.shape
        V, H2 = weight.shape
        assert H == H2, f"hidden H={H} != weight H={H2}"
        assert target.numel() == N

        # enable TF32 optionally (matmul only)
        orig_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)

        # We compute:
        #   loss_i = logsumexp_j (z_ij) - z_i,y_i
        # where z = hidden @ weight^T + bias
        # We do online logsumexp in blocks to avoid full z materialization.

        device = hidden_2d.device
        cdtype = compute_dtype

        # online logsumexp state: m (max) and s (sum exp)
        m = torch.full((N,), -float("inf"), device=device, dtype=cdtype)
        s = torch.zeros((N,), device=device, dtype=cdtype)
        true_logit = torch.zeros((N,), device=device, dtype=cdtype)

        # block loop
        for v0 in range(0, V, chunk_size):
            v1 = min(V, v0 + chunk_size)
            w_blk = weight[v0:v1]  # [vb, H]
            # logits block: [N, vb]
            # compute in hidden dtype (bf16/fp16) then cast to cdtype for reductions
            z = hidden_2d.matmul(w_blk.t())
            if bias is not None:
                z = z + bias[v0:v1]
            zf = z.to(cdtype)

            blk_max = zf.max(dim=1).values
            m_new = torch.maximum(m, blk_max)
            # rescale old sum, add new
            s = s * torch.exp(m - m_new) + torch.exp(zf - m_new.unsqueeze(1)).sum(dim=1)
            m = m_new

            # gather true logit if label falls into this block
            t = target
            in_blk = (t >= v0) & (t < v1)
            if in_blk.any():
                idx = torch.nonzero(in_blk, as_tuple=False).squeeze(1)
                # gather from zf: row idx, col (t - v0)
                col = (t[idx] - v0).to(torch.long)
                true_logit[idx] = zf[idx, col]

        logsumexp = torch.log(s) + m  # [N]
        loss_vec = logsumexp - true_logit  # [N]

        if reduction == "none":
            out = loss_vec
            scale = 1.0
        elif reduction == "sum":
            out = loss_vec.sum()
            scale = 1.0
        elif reduction == "mean":
            out = loss_vec.mean()
            scale = 1.0 / float(N)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        # save for backward
        # We need: hidden_2d, weight, target, bias, logsumexp, and scale
        ctx.save_for_backward(hidden_2d, weight, target, bias if bias is not None else torch.tensor([], device=device), logsumexp)
        ctx.chunk_size = int(chunk_size)
        ctx.reduction = reduction
        ctx.compute_dtype = compute_dtype
        ctx.allow_tf32 = bool(allow_tf32)
        ctx.has_bias = bias is not None
        ctx.scale = float(scale)
        ctx.V = int(V)
        # restore tf32 flag
        torch.backends.cuda.matmul.allow_tf32 = orig_tf32
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        hidden_2d, weight, target, bias_saved, logsumexp = ctx.saved_tensors
        bias = bias_saved if ctx.has_bias else None

        chunk_size = ctx.chunk_size
        cdtype = ctx.compute_dtype
        allow_tf32 = ctx.allow_tf32
        scale = ctx.scale
        V = ctx.V

        # If out is scalar (mean/sum), grad_out is scalar.
        # For reduction="none", grad_out is [N].
        if ctx.reduction == "none":
            # each row has its own upstream grad
            g_row = grad_out.to(cdtype)  # [N]
        else:
            g_row = grad_out.to(cdtype) * scale  # scalar

        # allow TF32 optionally (matmul only)
        orig_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)

        N, H = hidden_2d.shape
        device = hidden_2d.device

        # grads to return
        grad_hidden = torch.zeros_like(hidden_2d)
        # (optional) grad_weight/grad_bias
        # NOTE: In many of our experiments we detach weight to avoid training head weights.
        # In that case, autograd will ignore grad_weight anyway.
        grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
        grad_bias = torch.zeros((V,), device=device, dtype=cdtype) if (bias is not None and bias.requires_grad) else None

        # We need softmax(z) = exp(z - logsumexp)
        # And grad logits = softmax - onehot.
        # Then:
        #   grad_hidden += grad_logits @ weight
        #   grad_weight += grad_logits^T @ hidden
        #   grad_bias += grad_logits.sum(dim=0)

        # convert logsumexp to cdtype
        lse = logsumexp.to(cdtype)  # [N]

        # For each vocab block:
        #   z_blk = hidden @ w_blk^T (+bias)
        #   p_blk = exp(z_blk - lse)
        #   grad_blk = p_blk
        #   grad_blk[row, y-v0] -= 1
        #   grad_blk *= g_row (scalar or per-row)
        for v0 in range(0, V, chunk_size):
            v1 = min(V, v0 + chunk_size)
            w_blk = weight[v0:v1]  # [vb, H]
            z = hidden_2d.matmul(w_blk.t())
            if bias is not None:
                z = z + bias[v0:v1]
            zf = z.to(cdtype)

            # softmax block
            p = torch.exp(zf - lse.unsqueeze(1))  # [N, vb], cdtype
            # subtract onehot for true class within this block
            in_blk = (target >= v0) & (target < v1)
            if in_blk.any():
                idx = torch.nonzero(in_blk, as_tuple=False).squeeze(1)
                col = (target[idx] - v0).to(torch.long)
                p[idx, col] -= 1.0

            # scale by upstream grad
            if ctx.reduction == "none":
                p = p * g_row.unsqueeze(1)  # [N, vb]
            else:
                p = p * g_row  # scalar

            # grad_hidden: [N, H] += [N, vb] @ [vb, H]
            # do matmul in weight dtype for speed, but accumulation in fp32 via cdtype -> cast.
            grad_hidden = grad_hidden + p.to(hidden_2d.dtype).matmul(w_blk.to(hidden_2d.dtype))

            # grad_weight (optional)
            if grad_weight is not None:
                # [vb, H] += [vb, N] @ [N, H]
                gw = p.t().to(hidden_2d.dtype).matmul(hidden_2d.to(hidden_2d.dtype))
                grad_weight[v0:v1] = grad_weight[v0:v1] + gw

            if grad_bias is not None:
                grad_bias[v0:v1] = grad_bias[v0:v1] + p.sum(dim=0)

        # restore tf32 flag
        torch.backends.cuda.matmul.allow_tf32 = orig_tf32

        # return grads for inputs: hidden, weight, target, bias, and the rest None
        # target has no grad
        if grad_bias is not None:
            grad_bias = grad_bias.to(bias.dtype)
        return grad_hidden, grad_weight, None, (grad_bias if ctx.has_bias else None), None, None, None, None


def chunked_ce_loss(hidden_2d: torch.Tensor,
                    weight: torch.Tensor,
                    target: torch.Tensor,
                    bias: Optional[torch.Tensor] = None,
                    chunk_size: int = 8192,
                    reduction: str = "mean",
                    compute_dtype: torch.dtype = torch.float32,
                    allow_tf32: bool = True) -> torch.Tensor:
    """
    Exact cross-entropy loss with block-wise vocab processing.
    hidden_2d: [N, H] CUDA
    weight:    [V, H] CUDA
    target:    [N] long

    Returns:
      loss (scalar if reduction != "none", else [N])
    """
    return _ChunkedCrossEntropy.apply(hidden_2d, weight, target, bias, int(chunk_size), reduction, compute_dtype, allow_tf32)
