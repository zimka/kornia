r"""
Implementation of "sampling-argmax" operation, which improves soft-argmax with "implicit
constraints to the shape of the probability map" as described in the paper
"Localization with sampling-argmax." by Li, Jiefeng, et al. (https://arxiv.org/pdf/2110.08825.pdf)
"""
from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F

from kornia.core import Tensor, tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.utils.grid import create_meshgrid

from .dsnt import spatial_softmax2d


def _sample_heamaps_pihat(
    heatmap: Tensor, temperature: Tensor = tensor(1.0), num_samples: int = 5, is_training: bool = True
) -> Tensor:
    r"""
    Samples heatmaps with Gumbel-Softmax Trick (called \hat{pi} in the paper)

    Args:
        heatmap: the given heatmap with shape :math:`(B, N, H, W)`. Heatmap should NOT be
            normalized to 1, i.e. they should be "logits", not "probabilities"
        temperature: factor to apply to .
        num_samples: number of heatamaps to sample
    Returns:
        Sampled heatmaps :math:`(B, N, num_samples, H, W)`.
        Heatmaps are NOT normalized to 1, i.e. softmax is NOT applied to them here.
    """
    KORNIA_CHECK_IS_TENSOR(heatmap)
    KORNIA_CHECK_SHAPE(heatmap, ['B', 'C', 'H', 'W'])
    batch_size, channels, height, width = heatmap.shape
    input_broadcast = heatmap.reshape((batch_size, channels, 1, height, width))

    if not is_training:
        assert num_samples == 1
        return input_broadcast

    eps = torch.rand(batch_size, channels, num_samples, height, width, device=heatmap.device)
    gumbel_eps = torch.log(-torch.log(eps))

    pihat = (input_broadcast - gumbel_eps) / temperature
    return pihat


class ClipIntegral(torch.autograd.Function):
    """
    Modified operation to estimate heatmap first moment.
    Multiplies pi[i] with w[i] for `Y = \Sum{pi[i] * w[i]}`, but binarizes gradient
    for pi[i] to be either +1 or -1 instead of w[i].
    """

    AMPLITUDE = 2

    @staticmethod
    def forward(ctx, input, weight):
        assert isinstance(input, torch.Tensor), 'ClipIntegral only takes input as torch.Tensor'
        input_size = input.size()
        ctx.input_size = input_size
        output = input.mul(weight)
        ctx.save_for_backward(input, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, output = ctx.saved_tensors
        output_coord = output.sum(dim=-1, keepdim=True)
        weight = weight.expand_as(output)

        weight_mask = torch.ones(
            weight.shape, dtype=grad_output.dtype, layout=grad_output.layout, device=grad_output.device
        )
        weight_mask[weight < output_coord] = -1
        weight_mask *= ClipIntegral.AMPLITUDE
        return grad_output.mul(weight_mask), grad_output.mul(input)


def _estimate_1d_expectation(pihat, with_noise: bool = True):
    r"""
    Estimates first moment for given N 1d-distributions with D bins.
    """
    KORNIA_CHECK_IS_TENSOR(pihat)
    KORNIA_CHECK_SHAPE(pihat, ['N', 'D'])
    num_distribs, num_bins = pihat.shape

    yhat = torch.arange(num_bins, dtype=torch.float32, device=pihat.device).expand_as(pihat)

    if with_noise:
        # TODO: triangular and gaussian basis
        eps = torch.rand_like(yhat) - 0.5
        yhat = yhat + eps

    return ClipIntegral.apply(pihat, yhat).sum(axis=-1)


def spatial_sampling_argmax2d(
    input: Tensor,
    temperature: Tensor = tensor(1.0),
    normalized_coordinates: bool = True,
    num_samples: int = 10,
    is_training: bool = True,
) -> Tensor:
    r"""Draws spatial samples from a given input heatmap.

    Args:
        input: the given heatmap with shape :math:`(B, C, H, W)`.
        temperature: factor to apply to input.
        normalized_coordinates: whether to return the coordinates normalized in the range of
            :math:`[-1, 1]`. Otherwise, it will return the coordinates in the range of the input shape.
        num_samples: number of samples to get from heatmap distribution
        is_training: whether to use training-phase behavior or test-phase
    Returns:
        Samples of 2d coordinates of the given map :math:`(B, C, num_samples, 2)`.
        The output order is x-coord and y-coord.

    Examples:
        ??????
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    if not is_training:
        if num_samples != 1:
            raise ValueError(
                "num_samples must be 1 when not training.\
                Gumbel-Softmax Trick is not used in test-phase,\
                single sample without noise is used instead."
            )

    batch_size, channels, height, width = input.shape

    pihat_heatmaps = _sample_heamaps_pihat(
        heatmap=input, temperature=temperature, num_samples=num_samples if is_training else 1, is_training=is_training
    )
    assert pihat_heatmaps.shape == (batch_size, channels, num_samples, height, width)
    pihat_heatmaps_normed = spatial_softmax2d(
        pihat_heatmaps.view(batch_size, channels * num_samples, height, width), temperature=temperature
    ).view(pihat_heatmaps.shape)

    pihat_y = pihat_heatmaps_normed.sum(axis=-1)
    pihat_x = pihat_heatmaps_normed.sum(axis=-2)

    coords_y = _estimate_1d_expectation(pihat_y.view(batch_size * channels * num_samples, height), with_noise=is_training)
    coords_x = _estimate_1d_expectation(pihat_x.view(batch_size * channels * num_samples, width), with_noise=is_training)

    if normalized_coordinates:
        coords_y = coords_y / (height - 1) - 1
        coords_x = coords_x / (width - 1) - 1

    coords = torch.stack([coords_x, coords_y], dim=-1)
    return coords.view((batch_size, channels, num_samples, 2))
