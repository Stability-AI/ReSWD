import cv2
import numpy as np
import torch
from jaxtyping import Float


def read_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        img = img[..., np.newaxis]
    dinfo = np.iinfo(img.dtype)
    return (img.astype(np.float32) / dinfo.max) * 2 - 1


def write_img(path: str, data: np.ndarray):
    data = np.clip(data * 0.5 + 0.5, 0, 1)
    if data.ndim == 3 and data.shape[-1] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    elif data.ndim == 2:
        data = data[..., np.newaxis]

    data = (data * 255).astype(np.uint8)
    cv2.imwrite(path, data)


def to_torch(img: Float[np.ndarray, "H W C"]) -> Float[torch.Tensor, "C H W"]:
    return torch.from_numpy(img).permute(2, 0, 1)


def from_torch(img: Float[torch.Tensor, "C H W"]) -> Float[np.ndarray, "H W C"]:
    return img.permute(1, 2, 0).detach().cpu().float().numpy()
