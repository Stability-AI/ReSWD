import argparse
import os

import imageio
import torch


def main():
    """
    Entry point for the ReSWD CLI.

    Parses the main mode argument and dispatches to the appropriate subcommand:
    - colormatch: Color matches the input images to a target image.
    - swguidance: Generates images using SW Guidance with a reference image.

    Additional arguments for each mode are parsed and handled by their respective
    functions.
    """

    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(
        description=(
            "ReSWD: Sliced Wasserstein Distance with Weighted Reservoir Sampling "
            "for color matching or diffusion guidance.\n"
            "Choose a mode and see the help for each mode for more options."
        )
    )

    parser.add_argument(
        "mode",
        choices=[
            "colormatch",
            "swguidance",
        ],
        help=(
            "Selects the mode ReSWD is run in:\n"
            "  colormatch    - Color matches the input images to a target image.\n"
            "  swguidance    - Generates images using SW Guidance with a reference.\n"
        ),
    )

    mode_args, remaining = parser.parse_known_args()

    if mode_args.mode == "colormatch":
        colormatch(remaining)
    elif mode_args.mode == "swguidance":
        swguidance(remaining)


def base_args(
    parser,
    steps: int = 300,
    lr: float = 0.01,
    num_projections: int = 64,
    num_new_candidates: int = 16,
    precision: str = "16-mixed",
):
    """
    Adds the common arguments for all Tessera modes to the given parser.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.
        steps (int): Default number of optimization steps.
        lr (float): Default learning rate.
        num_projections (int): Default number of projections for SWD.
        num_new_candidates (int): Default number of new candidates for the reservoir
            in SWD.
        precision (str): Default precision for the computation.
    """
    parser.add_argument(
        "--precision",
        choices=["16-mixed", "32-true"],
        default=precision,
        help=(
            "Sets the computation precision mode.\n"
            "  16-mixed: Mixed 16-bit/32-bit precision (recommended for modern GPUs).\n"
            "  32-true:  Full 32-bit precision."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=lr,
        help=(
            "Learning rate for the optimizer. "
            "Lower values may yield more stable results but slower convergence. "
            f"Default: {lr}."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=steps,
        help=(
            "Number of optimization steps per scale. "
            "Increasing this may improve quality but increases runtime. "
            f"Default: {steps}."
        ),
    )
    parser.add_argument(
        "--num_projections",
        type=int,
        default=num_projections,
        help=(
            "Number of random projections for PatchSWDLoss. "
            "Higher values are more accurate but slower and use more memory. "
            f"Default: {num_projections}."
        ),
    )
    parser.add_argument(
        "--num_new_candidates",
        type=int,
        default=num_new_candidates,
        help=(
            "Number of new candidates for the reservoir in PatchSWDLoss. "
            "A value of 0 disables reservoir sampling. "
        ),
    )

    cv_group = parser.add_mutually_exclusive_group()
    cv_group.add_argument(
        "--use_ucv",
        action="store_true",
        dest="use_ucv",
        help=(
            "Use the UCV (Upper-bound Control Variate) variant of PatchSWDLoss. "
            "Mutually exclusive with LCV. If neither is set, no control variates are "
            "used."
        ),
    )
    cv_group.add_argument(
        "--use_lcv",
        action="store_true",
        help=(
            "Use the LCV (Lower-bound Control Variate) variant of PatchSWDLoss. "
            "Mutually exclusive with UCV. If neither is set, no control variates are "
            "used."
        ),
    )

    parser.add_argument(
        "--distance",
        choices=["l1", "l2"],
        default="l1",
        help=(
            "Distance metric for PatchSWDLoss:\n"
            "  l1: Manhattan (absolute value) distance.\n"
            "  l2: Euclidean (squared) distance.\n"
            "Default: l1."
        ),
    )

    parser.add_argument(
        "--refresh_projections_every_n_steps",
        type=int,
        default=1,
        help=(
            "How often (in steps) to refresh the random projections in PatchSWDLoss. "
            "Lower values may improve diversity but increase runtime. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--sampling_mode",
        choices=["gaussian", "qmc"],
        default="gaussian",
        help=("Sampling mode for PatchSWDLoss. Default: gaussian."),
    )


def colormatch(remaining_args):
    """
    Handles the 'colormatch' mode: color matches the input images to a target image.

    Args:
        remaining_args (list): List of command-line arguments for this mode.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Color matches the input images to a target image. "
            "It will create a CDL parameter fit and apply it to the input images."
        )
    )

    parser.add_argument("save_dir", type=str, help="Directory to save the results to.")
    parser.add_argument(
        "--source_images",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Paths to the source images to color match. "
            "These images are assumed to belong in pairs (img 1 of source_images is "
            "matched to img 1 of target_images, etc.)."
        ),
    )
    parser.add_argument(
        "--target_images",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Paths to the target images to color match. "
            "These images are assumed to belong in pairs (img 1 of source_images is "
            "matched to img 1 of target_images, etc.)."
        ),
    )
    parser.add_argument(
        "--matching_resolution",
        type=int,
        default=128,
        help="Resolution to match the input images to. Default: 128.",
    )
    parser.add_argument(
        "--write_video",
        action="store_true",
        help="Write a video of the color matching process.",
    )

    base_args(parser, steps=200)

    args = parser.parse_args(remaining_args)

    if len(args.source_images) != len(args.target_images):
        raise ValueError(
            "The number of source images must match the number of target images."
        )

    from src.color_matcher import run

    run(
        args.save_dir,
        args.source_images,
        args.target_images,
        matching_resolution=args.matching_resolution,
        precision=args.precision,
        num_projections=args.num_projections,
        lr=args.lr,
        steps=args.steps,
        use_ucv=args.use_ucv,
        use_lcv=args.use_lcv,
        distance=args.distance,
        refresh_projections_every_n_steps=args.refresh_projections_every_n_steps,
        num_new_candidates=args.num_new_candidates,
        sampling_mode=args.sampling_mode,
        write_video=args.write_video,
    )


def swguidance(remaining_args):
    """
    Handles the 'swguidance' mode: generates images using SW Guidance with a
    reference image.

    Args:
        remaining_args (list): List of command-line arguments for this mode.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generates images using SW Guidance with a reference image. "
            "Use this mode to create images that follow a text prompt while "
            "maintaining the style and appearance of a reference image."
        )
    )

    parser.add_argument("save_dir", type=str, help="Directory to save the results to.")
    parser.add_argument(
        "--model_path",
        type=str,
        help=(
            "Path to the model weights (.safetensors file). If not specified, will use "
            "the default path for the selected model."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to guide the generation.",
    )
    parser.add_argument(
        "--reference_image",
        type=str,
        required=True,
        help="Path to the reference image to guide the generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps. Default: 30.",
    )
    parser.add_argument(
        "--num_guided_steps_perc",
        type=float,
        default=0.95,
        help="Percentage of steps to apply SW guidance. Default: 0.95.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Scale for classifier-free guidance. Default: 3.5.",
    )
    parser.add_argument(
        "--sw_u_lr",
        type=float,
        default=3e-3,
        help="Learning rate for SW guidance. Default: 3e-3.",
    )
    parser.add_argument(
        "--sw_steps",
        type=int,
        default=6,
        help="Number of steps to apply SW guidance. Default: 6.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Output image height. Default: 1024.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Output image width. Default: 1024.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on. Default: cuda.",
    )

    # Add SW-related parameters
    parser.add_argument(
        "--num_projections",
        type=int,
        default=32,
        help="Number of random projections for VectorSWDLoss. Default: 32.",
    )
    parser.add_argument(
        "--use_ucv",
        action="store_true",
        help=(
            "Use the UCV (Upper-bound Control Variate) variant of SWDLoss. "
            "Mutually exclusive with LCV. If neither is set, no control variates are "
            "used."
        ),
    )
    parser.add_argument(
        "--use_lcv",
        action="store_true",
        help=(
            "Use the LCV (Lower-bound Control Variate) variant of SWDLoss. "
            "Mutually exclusive with UCV. If neither is set, no control variates are "
            "used."
        ),
    )
    parser.add_argument(
        "--distance",
        choices=["l1", "l2"],
        default="l1",
        help=(
            "Distance metric for SWDLoss:\n"
            "  l1: Manhattan (absolute value) distance.\n"
            "  l2: Euclidean (squared) distance.\n"
            "Default: l1."
        ),
    )
    parser.add_argument(
        "--num_new_candidates",
        type=int,
        default=0,
        help=(
            "Number of new candidates for the reservoir in SWDLoss. "
            "A value of 0 disables reservoir sampling. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--sampling_mode",
        choices=["gaussian", "qmc"],
        default="gaussian",
        help=("Sampling mode for PatchSWDLoss. Default: gaussian."),
    )
    parser.add_argument(
        "--subsampling_factor",
        type=int,
        default=1,
        help=(
            "Factor to subsample points for SW computation. Higher values reduce "
            "memory usage but may affect quality. For example, a value of 2 will use "
            "every second point. Default: 1 (no subsampling)."
        ),
    )
    parser.add_argument(
        "--write_video",
        action="store_true",
        help="Write a video of the image generation process.",
    )

    args = parser.parse_args(remaining_args)

    # Set default model path based on selection
    if args.model_path is None:
        args.model_path = "stabilityai/stable-diffusion-3.5-large"

    if args.height is None:
        args.height = 1024
    if args.width is None:
        args.width = 1024

    from PIL import Image

    from src.sw_sdthree_guidance import run as sw_guidance_run

    # Load reference image
    ref_image = Image.open(args.reference_image).convert("RGB")

    animation_dir = os.path.join(args.save_dir, "animation")
    if args.write_video:
        os.makedirs(animation_dir, exist_ok=True)

    # Generate image
    result = sw_guidance_run(
        prompt=args.prompt,
        reference_image=ref_image,
        model_path=args.model_path,
        num_inference_steps=args.num_inference_steps,
        num_guided_steps=int(args.num_guided_steps_perc * args.num_inference_steps),
        guidance_scale=args.guidance_scale,
        sw_u_lr=args.sw_u_lr,
        sw_steps=args.sw_steps,
        height=args.height,
        width=args.width,
        device=args.device,
        # Add new SW-related parameters
        num_projections=args.num_projections,
        use_ucv=args.use_ucv,
        use_lcv=args.use_lcv,
        distance=args.distance,
        num_new_candidates=args.num_new_candidates,
        subsampling_factor=args.subsampling_factor,
        sampling_mode=args.sampling_mode,
        video_animation_path=animation_dir if args.write_video else None,
    )

    if args.write_video:
        # Get the list of image files in the animation directory
        image_files = [f for f in os.listdir(animation_dir) if f.endswith(".jpg")]
        image_files.sort(
            key=lambda x: int(x.split(".")[0])
        )  # Ensure they are in the correct order

        # Create a video from the images
        with imageio.get_writer(
            os.path.join(args.save_dir, "animation.mp4"), fps=30, codec="libx264"
        ) as writer:
            for image_file in image_files:
                image = imageio.imread(os.path.join(animation_dir, image_file))
                writer.append_data(image)

    # Save result
    os.makedirs(args.save_dir, exist_ok=True)
    result.save(os.path.join(args.save_dir, "output.jpg"))


if __name__ == "__main__":
    main()
