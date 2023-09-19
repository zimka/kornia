import torch
from torch.autograd import gradcheck
from torch.nn.functional import mse_loss

import kornia
import kornia.testing as utils  # test utils
from kornia.geometry.subpix.spatial_sampling_argmax import spatial_sampling_argmax2d
from kornia.testing import assert_close


class TestSamplingArgmax:

    def test_top_left_normalized(self, device, dtype):
        heatmap = torch.zeros(1, 1, 4, 5, device=device, dtype=dtype)
        heatmap[..., 0, 0] = 1e16
        num_samples = 20

        coord = spatial_sampling_argmax2d(
            heatmap, num_samples=num_samples, normalized_coordinates=False
        )
        assert coord.shape == (1, 1, num_samples, 2)
        zeros = torch.zeros_like(coord[..., 0]) + 1e-5
        assert_close(coord[..., 0], zeros, atol=0.5, rtol=float('inf'))
        assert_close(coord[..., 1], zeros, atol=0.5, rtol=float('inf'))

        assert heatmap.shape == (1, 1, 4, 5)
        coord = spatial_sampling_argmax2d(
            heatmap, num_samples=1, is_training=False, normalized_coordinates=False
        )
        assert coord.shape == (1, 1, 1, 2)
        zeros = torch.zeros_like(coord[..., 0]) + 1e-5
        assert_close(coord[..., 0], zeros, atol=1e-4, rtol=1e-4)
        assert_close(coord[..., 1], zeros, atol=1e-4, rtol=1e-4)
