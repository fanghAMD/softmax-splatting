from pathlib import Path
from typing import Literal

import cv2
import flow_vis
import numpy as np
import pyexr
import tifffile
import torch


class MedianPool2d(torch.nn.Module):
    """
    Median pool (usable as median filter when stride=1) module.
    (https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598)

    Args:
        kernel_size: size of pooling kernel, int or 2-tuple
        stride: pool stride, int or 2-tuple
        padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
        same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        from torch.nn.modules.utils import _pair, _quadruple

        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = torch.nn.functional.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def to_uint8_image(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float or t.dtype == torch.half:
        t = (t.clip(0, 1) * 255).to(torch.uint8)
    else:
        assert t.dtype == torch.uint8
    return t


def bgr_to_ycocg(bgr: torch.Tensor) -> torch.Tensor:
    b = bgr[:, 0:1]
    g = bgr[:, 1:2]
    r = bgr[:, 2:3]
    return torch.cat(
        [
            r / 4 + g / 2 + b / 4,
            -r / 4 + g / 2 - b / 4,
            r / 2 - b / 2,
        ],
        dim=1,
    )


def srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    t = t.clip(0, 1)
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055).pow(2.4))


def visualize_flow(flow: torch.Tensor, batch: int = 0) -> torch.Tensor:
    flow_np = flow[batch].permute(1, 2, 0).cpu().numpy()
    color_np = flow_vis.flow_to_color(flow_np, convert_to_bgr=True)
    color = torch.from_numpy(color_np).permute(2, 0, 1).unsqueeze(0)
    return color.to(device=flow.device)


def read_png(file: str | Path) -> torch.Tensor:
    bgr_np = cv2.imread(str(file))
    bgr = torch.from_numpy(bgr_np)
    return bgr.permute(2, 0, 1).unsqueeze(dim=0)


def write_png(file: str | Path, bgr: torch.Tensor, batch: int = 0) -> None:
    Path(file).parent.mkdir(parents=True, exist_ok=True)

    # TODO: RGB, uint16, ...
    bgr = to_uint8_image(bgr[batch])

    cv2.imwrite(str(file), bgr.permute(1, 2, 0).cpu().detach().numpy())


def read_tiff(file: str | Path) -> torch.Tensor:
    img_np = tifffile.imread(file)

    if img_np.dtype == np.uint32:
        img_np = img_np.astype(np.int64)

    img = torch.from_numpy(img_np)

    if img.ndim == 2:
        img.unsqueeze(dim=-1)
    assert img.ndim == 3

    return img.permute(2, 0, 1).unsqueeze(dim=0)


def write_tiff(file: str | Path, img: torch.Tensor, batch: int = 0) -> None:
    Path(file).parent.mkdir(parents=True, exist_ok=True)

    img_np = img[batch].permute(1, 2, 0).cpu().detach().numpy()

    assert isinstance(img_np, np.ndarray)

    if img_np.dtype == np.int64:
        img_np = img_np.astype(np.uint32)

    tifffile.imwrite(file, img_np, compression="zlib")


def read_exr(
    file: str | Path,
    channels: str = "default",
    precision: Literal["half", "float", "uint"] = "float",
):
    str_to_pixtype = {"half": pyexr.HALF, "float": pyexr.FLOAT, "uint": pyexr.UINT}
    img_np = pyexr.read(str(file), channels, precision=str_to_pixtype[precision])

    assert isinstance(img_np, np.ndarray)

    if img_np.dtype == np.uint32:
        img_np = img_np.astype(np.int64)

    img = torch.from_numpy(img_np)

    if img.ndim == 2:
        img.unsqueeze(dim=-1)
    assert img.ndim == 3

    return img.permute(2, 0, 1).unsqueeze(dim=0)


def write_exr(
    file: str | Path,
    img: torch.Tensor,
    batch: int = 0,
    compression: Literal["none", "rle", "zips", "zip", "piz", "pxr24"] = "none",
):
    Path(file).parent.mkdir(parents=True, exist_ok=True)

    img_np = img[batch].permute(1, 2, 0).cpu().detach().numpy()

    assert isinstance(img_np, np.ndarray)

    if img_np.dtype == np.int64:
        img_np = img_np.astype(np.uint32)

    dtype_to_pixtype = {
        "float16": pyexr.HALF,
        "float32": pyexr.FLOAT,
        "uint32": pyexr.UINT,
    }

    str_to_compression = {
        "none": pyexr.NO_COMPRESSION,
        "rle": pyexr.RLE_COMPRESSION,
        "zips": pyexr.ZIPS_COMPRESSION,
        "zip": pyexr.ZIP_COMPRESSION,
        "piz": pyexr.PIZ_COMPRESSION,
        "pxr24": pyexr.PXR24_COMPRESSION,
    }

    pyexr.write(
        str(file),
        img_np,
        precision=dtype_to_pixtype[img_np.dtype.name],
        compression=str_to_compression[compression],
    )
