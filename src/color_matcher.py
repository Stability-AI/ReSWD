import gc
import os
from typing import List, Literal, Optional, Tuple

import imageio
import lightning as L
import torch
from jaxtyping import Float
from torchvision.transforms import Resize
from tqdm import tqdm

from src.loss import AbstractLoss
from src.loss.vector_swd import VectorSWDLoss
from src.utils.asc_cdl import asc_cdl_forward, save_asc_cdl
from src.utils.color_space import rgb_to_lab
from src.utils.image import from_torch, read_img, to_torch, write_img


class CDL(torch.nn.Module):
    def __init__(self, batch_size: int):
        super().__init__()
        self.cdl_slope = torch.nn.Parameter(torch.ones(batch_size, 3))
        self.cdl_offset = torch.nn.Parameter(torch.zeros(batch_size, 3))
        self.cdl_power = torch.nn.Parameter(torch.ones(batch_size, 3))
        self.cdl_saturation = torch.nn.Parameter(torch.ones(batch_size))

    def forward(
        self, x: Float[torch.Tensor, "*B C H W"]
    ) -> Float[torch.Tensor, "*B C H W"]:
        return asc_cdl_forward(
            x, self.cdl_slope, self.cdl_offset, self.cdl_power, self.cdl_saturation
        )

    def to_cdl_xml(self) -> str:
        ret = []
        for b in range(self.cdl_slope.shape[0]):
            ret.append(
                save_asc_cdl(
                    {
                        "slope": self.cdl_slope[b],
                        "offset": self.cdl_offset[b],
                        "power": self.cdl_power[b],
                        "saturation": self.cdl_saturation[b],
                    },
                    None,
                )
            )
        return ret

    def save(self, path: str):
        for b in range(self.cdl_slope.shape[0]):
            save_asc_cdl(
                {
                    "slope": self.cdl_slope[b],
                    "offset": self.cdl_offset[b],
                    "power": self.cdl_power[b],
                    "saturation": self.cdl_saturation[b],
                },
                os.path.join(path, f"cdl_{b}.xml"),
            )


def train(
    fabric: L.Fabric,
    criteria: AbstractLoss,
    source_img: Float[torch.Tensor, "B C H W"],
    target_img: Float[torch.Tensor, "B C H W"],
    num_steps: int,
    lr: float,
    match_resolution: int,
    silent: bool = False,
    write_video_animation_path: Optional[str] = None,
) -> Tuple[Float[torch.Tensor, "*B C H W"], CDL, List[float]]:
    criteria = fabric.setup(criteria)

    source_max_res = Resize(match_resolution, antialias=True)(source_img)
    target_max_res = Resize(match_resolution, antialias=True)(target_img)

    target_cielab = (
        fabric.to_device(rgb_to_lab(target_max_res).permute(0, 3, 1, 2))
        .permute(0, 2, 3, 1)
        .contiguous()
    )

    source_max_res = fabric.to_device(source_max_res)
    source_img = fabric.to_device(source_img)

    batch_size = source_img.shape[0]
    cdl = CDL(batch_size)

    optim = torch.optim.Adam(cdl.parameters(), lr=lr)
    cdl, optim = fabric.setup(cdl, optim)

    lossses = []
    for i in tqdm(range(num_steps), disable=silent):
        optim.zero_grad(set_to_none=True)

        cdl_source = cdl(source_max_res)
        source_cielab = (
            rgb_to_lab(cdl_source.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        )

        loss = criteria(
            source_cielab.view(source_cielab.shape[0], source_cielab.shape[1], -1),
            target_cielab.view(target_cielab.shape[0], target_cielab.shape[1], -1),
            i,
        )

        fabric.backward(loss)
        optim.step()

        lossses.append(loss.item())

        if write_video_animation_path is not None:
            write_img(
                os.path.join(write_video_animation_path, f"{i:05d}.jpg"),
                from_torch(cdl(source_img).squeeze(0) * 2 - 1),
            )

    source_full_res_cdl = cdl(source_img)

    gc.collect()
    torch.cuda.empty_cache()

    return source_full_res_cdl, cdl, lossses


def run(
    save_dir: str,
    source_img: List[str],
    target_img: List[str],
    matching_resolution: int,
    precision: Literal["32-true", "16-mixed"] = "16-mixed",
    num_projections: int = 64,
    lr: float = 0.01,
    steps: int = 300,
    use_ucv: bool = False,
    use_lcv: bool = False,
    distance: Literal["l1", "l2"] = "l1",
    refresh_projections_every_n_steps: int = 1,
    num_new_candidates: int = 32,
    sampling_mode: Literal["gaussian", "qmc"] = "gaussian",
    write_video: bool = False,
    **kwargs,
):
    fabric = L.Fabric(devices=1, accelerator="auto", precision=precision)

    source_imgs = torch.stack(
        [to_torch(read_img(s)) * 0.5 + 0.5 for s in source_img], dim=0
    )
    target_imgs = torch.stack(
        [to_torch(read_img(t)) * 0.5 + 0.5 for t in target_img], dim=0
    )

    criteria = VectorSWDLoss(
        num_proj=num_projections,
        distance=distance,
        use_ucv=use_ucv,
        use_lcv=use_lcv,
        refresh_projections_every_n_steps=refresh_projections_every_n_steps,
        num_new_candidates=num_new_candidates,
        sampling_mode=sampling_mode,
    )

    os.makedirs(save_dir, exist_ok=True)
    animation_dir = os.path.join(save_dir, "animation")

    if write_video:
        os.makedirs(animation_dir, exist_ok=True)

    source_full_res_cdl, cdl, lossses = train(
        fabric,
        criteria,
        source_imgs,
        target_imgs,
        steps,
        lr,
        matching_resolution,
        write_video_animation_path=animation_dir if write_video else None,
    )

    cdl.save(save_dir)

    for i, img in enumerate(source_full_res_cdl):
        write_img(
            os.path.join(save_dir, f"color_matched_{i}.png"),
            from_torch(img * 2 - 1),
        )

    if write_video:
        # Get the list of image files in the animation directory
        image_files = [f for f in os.listdir(animation_dir) if f.endswith(".jpg")]
        image_files.sort(
            key=lambda x: int(x.split(".")[0])
        )  # Ensure they are in the correct order

        # Create a video from the images
        with imageio.get_writer(
            os.path.join(save_dir, "animation.mp4"), fps=30, codec="libx264"
        ) as writer:
            for image_file in image_files:
                image = imageio.imread(os.path.join(animation_dir, image_file))
                writer.append_data(image)

    return source_full_res_cdl, cdl, lossses
