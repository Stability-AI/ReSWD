import torch
from jaxtyping import Float


def srgb_to_linear(x: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    switch_val = 0.04045
    return torch.where(
        torch.greater(x, switch_val),
        ((x.clip(min=switch_val) + 0.055) / 1.055).pow(2.4),
        x / 12.92,
    )


def linear_to_srgb(x: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    switch_val = 0.0031308
    return torch.where(
        torch.greater(x, switch_val),
        1.055 * x.clip(min=switch_val).pow(1.0 / 2.4) - 0.055,
        x * 12.92,
    )


def rgb_to_lab(srgb: Float[torch.Tensor, "*B C"]) -> Float[torch.Tensor, "*B C"]:
    srgb_pixels = torch.reshape(srgb, [-1, 3])

    linear_mask = srgb_pixels <= 0.04045
    exponential_mask = srgb_pixels > 0.04045
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
        ((srgb_pixels + 0.055) / 1.055) ** 2.4
    ) * exponential_mask

    rgb_to_xyz = (
        torch.tensor(
            [
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ]
        )
        .to(srgb.dtype)
        .to(srgb.device)
    )

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)

    xyz_normalized_pixels = torch.mul(
        xyz_pixels,
        torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754]).to(srgb.dtype).to(srgb.device),
    )

    epsilon = 6.0 / 29.0
    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).to(srgb.dtype).to(srgb.device)

    exponential_mask = (
        (xyz_normalized_pixels > (epsilon**3)).to(srgb.dtype).to(srgb.device)
    )

    fxfyfz_pixels = (
        xyz_normalized_pixels / (3 * epsilon**2) + 4.0 / 29.0
    ) * linear_mask + (
        (xyz_normalized_pixels + 0.000001) ** (1.0 / 3.0)
    ) * exponential_mask

    fxfyfz_to_lab = (
        torch.tensor(
            [
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ]
        )
        .to(srgb.dtype)
        .to(srgb.device)
    )
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor(
        [-16.0, 0.0, 0.0]
    ).to(srgb.dtype).to(srgb.device)
    return torch.reshape(lab_pixels, srgb.shape)
