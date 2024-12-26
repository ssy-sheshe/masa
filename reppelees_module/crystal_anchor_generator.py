import warnings

# import mmcv
import numpy as np
import torch

from mmdet.registry import TASK_UTILS
from torch.nn.modules.utils import _pair


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # [scale_num]
    ws = w * scales
    # [scale_num]
    hs = h * scales
    # [scale_num, 4]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""

    w, h, x_ctr, y_ctr = _whctrs(anchor)

    size = w * h
    size_ratios = size / ratios
    # ws = np.round(np.sqrt(size_ratios))
    # hs = np.round(ws * ratios)

    ws = np.sqrt(size_ratios)
    hs = ws * ratios
    # print(np.sqrt(size_ratios), ws, ws * ratios, hs)

    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

    return anchors


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    # ws, hs: [3, 1] (djf)
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    # anchors: [3, 4] (djf)
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


@TASK_UTILS.register_module()
class CrystalAnchorGenerator():

    def __init__(
        self,
        anchor_strides=[8, 16, 32, 64],
        aspect_ratios=[1.0, 2.5, 3.5],
        anchor_sizes=[20, 40, 80, 160],
        octave_base_scale=2.0,
        scales_per_octave=2,
    ):

        assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"

        new_anchor_sizes = []
        for size in anchor_sizes:
            per_layer_anchor_sizes = []
            for scale_per_octave in range(scales_per_octave):
                # 2^(0/3), 2^(1/3), 2^(2/3) djf
                octave_scale = octave_base_scale ** (scale_per_octave / float(scales_per_octave))
                per_layer_anchor_sizes.append(octave_scale * size)
            new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

        self.anchor_sizes = new_anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.anchor_strides = anchor_strides
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave

        self.cell_anchors = self.generate_cell_anchors(self.anchor_sizes, self.aspect_ratios, self.anchor_strides)
        
        # self.strides = self.anchor_strides
        self.strides = [_pair(stride) for stride in anchor_strides]

    def generate_cell_anchors(self, sizes, aspect_ratios, anchor_strides):
        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                torch.from_numpy(self.generate_anchors(anchor_stride, sizes, aspect_ratios)).float()
            ]
        else:

            cell_anchors = [
                torch.from_numpy(self.generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                )).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        return cell_anchors


    def generate_anchors(self, stride=16, sizes=(20.0, 28.28), aspect_ratios=(0.5, 1, 2)):
        """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
        are centered on stride / 2, have (approximate) sqrt areas of the specified
        sizes, and aspect ratios as given.
        """
        _base_size = stride
        _scales = np.array(sizes, dtype=float) / stride
        _aspect_ratios = np.array(aspect_ratios, dtype=float)

        anchor = np.array([1, 1, _base_size, _base_size], dtype=float) - 1
        # print(anchor)
        anchors = _ratio_enum(anchor, _aspect_ratios)
        # print(anchors)
        anchors = np.vstack([_scale_enum(anchors[i, :], _scales) for i in range(anchors.shape[0])])
        # print(anchors)
        # print('================')
        return anchors


    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors


    @property
    def num_base_priors(self):
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [cell_anchors.size(0) for cell_anchors in self.cell_anchors]


    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.anchor_strides)

    
    def single_level_grid_priors(self, featmap_size, level_idx, dtype=torch.float32, device='cuda'):

        # size = featmap_size[level_idx]
        stride = self.anchor_strides[level_idx]
        # print(self.cell_anchors[level_idx])
        base_anchors = self.cell_anchors[level_idx].to(device).to(dtype)
        # grid_height, grid_width = size
        grid_height, grid_width = featmap_size
        shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=dtype, device=device)
        shifts_y = torch.arange(0, grid_height * stride, step=stride, dtype=dtype, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
        return anchors


    def grid_priors(self, featmap_sizes, dtype=torch.float32, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            dtype (:obj:`torch.dtype`): Dtype of priors.
                Default: torch.float32.
            device (str): The device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors
    

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.anchor_strides[i]
            if not isinstance(anchor_stride, int):
                raise NotImplemented
            
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags
    
    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        # valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)

        valid_xx = valid_x.repeat(valid_y.shape[0])
        valid_yy = valid_y.view(-1, 1).repeat(1, valid_x.shape[0]).view(-1)

        # if row_major:
        #     return xx, yy
        # else:
        #     return yy, xx

        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid


if __name__ == '__main__':
    
    c = CrystalAnchorGenerator(
            anchor_sizes=[20, 40, 80, 160],
            octave_base_scale=2.0,
            scales_per_octave=2,
            aspect_ratios=[1.0, 2.5, 3.5],
            anchor_strides=[8, 16, 32, 64])
    
    print(c.cell_anchors)

    # anchors = c.grid_priors(featmap_sizes=[[72,128], [36,64], [18, 32], [9, 16]])
