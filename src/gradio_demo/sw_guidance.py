import os
from typing import Optional

import gradio as gr
import lightning as L
import numpy as np
from PIL import Image

from src.sw_sdthree_guidance import create_pipeline
from src.sw_sdthree_guidance import run as sw_guidance_run

preset_lrs = [1e-6, 1.0]

log_lrs = np.log10(preset_lrs)

models = {
    "stabilityai/stable-diffusion-3.5-large": {
        "num_inference_steps": 30,
        "guidance_scale": 7.0,
        "sw_u_lr": np.log10(3.2e-3),
        "sw_steps": 6,
        "cfg_rescale_phi": 0.7,
    },
    "stabilityai/stable-diffusion-3.5-medium": {
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "sw_u_lr": np.log10(3.2e-3),
        "sw_steps": 6,
        "cfg_rescale_phi": 0.7,
    },
    "stabilityai/stable-diffusion-3.5-large-turbo": {
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "sw_u_lr": np.log10(4e-3),
        "sw_steps": 6,
        "cfg_rescale_phi": 0.65,
    },
}


def log_slider_to_lr(log_lr):
    return float(f"{10**log_lr:.1e}")


def create_sw_guidance(
    fabric: L.Fabric, model_name: str = "stabilityai/stable-diffusion-3.5-large"
):
    """
    Creates the Gradio interface for SW guidance with SD3.5.

    Args:
        fabric: Lightning Fabric instance
        model_name: The model to use for guidance
    """
    gr.Markdown(
        f"""
        # SW Guidance with {model_name.split('/')[-1]}
        Generates images using SW Guidance with a reference image and text prompt.
        """
    )

    pipe = create_pipeline(
        model_name,
        device="cuda",
        compile=True,
    )

    model_config = models[model_name]

    def run_sw_guidance(
        num_inference_steps: int,
        num_guided_steps_perc: float,
        guidance_scale: float,
        sw_u_lr: float,
        sw_steps: int,
        num_projections: int,
        control_variates: str,
        distance: str,
        candidates_per_pass: int,
        subsampling_factor: int,
        sampling_mode: str,
        cfg_rescale_phi: float,
        prompt: str,
        reference_image: np.ndarray,
        seed: Optional[int] = None,
    ):
        """
        Runs the SW guidance process with the given parameters.
        """
        if reference_image is None:
            raise gr.Error("Please provide a reference image")
        if not prompt:
            raise gr.Error("Please provide a prompt")

        # Convert numpy array to PIL Image
        ref_img = Image.fromarray(reference_image)

        # Run SW guidance
        image = sw_guidance_run(
            prompt=prompt,
            reference_image=ref_img,
            model_path=model_name,
            num_inference_steps=num_inference_steps,
            num_guided_steps=int(num_guided_steps_perc * num_inference_steps),
            guidance_scale=guidance_scale,
            sw_u_lr=log_slider_to_lr(sw_u_lr),
            sw_steps=sw_steps,
            height=1024,
            width=1024,
            device="cuda",
            num_projections=num_projections,
            use_ucv=control_variates == "Upper",
            use_lcv=control_variates == "Lower",
            distance=distance,
            num_new_candidates=candidates_per_pass,
            subsampling_factor=subsampling_factor,
            sampling_mode=sampling_mode,
            pipe=pipe,
            compile=True,
            seed=seed,
            cfg_rescale_phi=cfg_rescale_phi,
        )

        return np.array(image)

    gr.Markdown("## Input")
    with gr.Row(equal_height=True):
        with gr.Column(variant="panel"):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=2,
            )
            reference_image = gr.Image(label="Reference Image", height=512)

        with gr.Column(variant="panel"):
            output_image = gr.Image(label="Generated Image", height=512)

    example_pairs = [
        ("A diver discovering an underwater city", "sunken_boat.jpg"),
        ("A raccoon in a forest", "waterfall.jpg"),
        ("A cat detective solving a mystery", "building.jpg"),
        ("A raccoon reading a book by candlelight.", "food_stand.jpg"),
        ("A lion meditating on a mountain", "mountain.jpg"),
        ("A squirrel kayaking", "lake_green.jpg"),
        ("A young dragon roasting marshmallows", "canyon.jpg"),
        ("A family picnic beneath floating lanterns", "lake_sunset.jpg"),
        ("Elephants holding umbrellas in a rainstorm", "fisher.jpg"),
        ("A boy and his dog exploring a crystal cave", "lake.jpg"),
        ("A snowman sharing cocoa with woodland animals", "snow.jpg"),
        ("A kitten exploring an antique library", "ornament.jpg"),
        ("Children riding flying bicycles over mountains", "sky_pier.jpg"),
        ("An ancient tree whispering stories to deer", "greenhouse_2.jpg"),
        ("Owls with monocles in treetops", "path.jpg"),
    ]

    default_config = {
        "num_guided_steps_perc": 0.95,
        "num_projections": 32,
        "control_variates": "None",
        "distance": "l1",
        "candidates_per_pass": 8,
        "subsampling_factor": 1,
        "sampling_mode": "gaussian",
    } | model_config

    def run_example(prompt: str, reference_image: np.ndarray):
        return run_sw_guidance(
            **default_config,
            prompt=prompt,
            reference_image=reference_image,
        )

    example_inputs = [
        [prompt, os.path.join("example", "guidance", img_file)]
        for prompt, img_file in example_pairs
    ]

    run_button = gr.Button("Generate Image", variant="primary")

    gr.Examples(
        examples=example_inputs,
        inputs=[prompt, reference_image],
        outputs=[output_image],
        fn=run_example,
        label="Prompt + Reference Image Examples",
        examples_per_page=5,
        cache_examples=True,
        cache_mode="lazy",
    )

    gr.Markdown(
        """
        # Configuration
        Adjust the parameters for the SW guidance process.
        """
    )

    with gr.Accordion("Basic Config", open=True):
        num_inference_steps_slider = gr.Slider(
            1,
            100,
            value=model_config["num_inference_steps"],
            step=1,
            label="Number of inference steps",
        )

        guidance_scale_slider = gr.Slider(
            0.0,
            20.0,
            value=model_config["guidance_scale"],
            step=0.1,
            label="Guidance scale",
        )

        cfg_rescale_phi_slider = gr.Slider(
            0.0,
            1.0,
            value=model_config["cfg_rescale_phi"],
            step=0.05,
            label="CFG Rescale Phi",
            info="Controls the rescaling of classifier-free guidance",
        )

        seed_input = gr.Number(
            value=lambda: None,
            label="Seed (leave empty for random)",
            precision=0,
            interactive=True,
        )

        with gr.Row(variant="panel", equal_height=True):
            sw_u_lr_slider = gr.Slider(
                minimum=log_lrs.min(),
                maximum=log_lrs.max(),
                value=model_config["sw_u_lr"],
                step=0.05,
                label="SW guidance learning rate (Log scale)",
                interactive=True,
                scale=4,
            )
            lr_display = gr.Textbox(
                label="Learning Rate",
                value=f"{log_slider_to_lr(model_config['sw_u_lr']):.1e}",
                interactive=False,
                scale=1,
            )
        sw_u_lr_slider.change(
            lambda x: gr.update(value=f"{log_slider_to_lr(x):.1e}"),
            inputs=sw_u_lr_slider,
            outputs=lr_display,
            show_progress=False,
        )

    with gr.Accordion("Advanced Config", open=False):
        num_guided_steps_perc_slider = gr.Slider(
            0.0,
            1.0,
            value=default_config["num_guided_steps_perc"],
            step=0.05,
            label="Percentage of steps to apply SW guidance",
        )
        sw_steps_slider = gr.Slider(
            0,
            32,
            value=model_config["sw_steps"],
            step=1,
            label="Number of SW guidance steps, 0 means no SW guidance",
        )
        num_projections_slider = gr.Slider(
            16,
            1024,
            value=default_config["num_projections"],
            step=16,
            label="Number of projections",
        )
        control_variates_dropdown = gr.Dropdown(
            choices=["None", "Lower", "Upper"],
            value=default_config["control_variates"],
            label="Control Variates",
            info="Select which control variates to use for optimization",
        )
        distance_dropdown = gr.Dropdown(
            choices=["l1", "l2"],
            value=default_config["distance"],
            label="Distance metric",
            info="Select which distance metric to use",
        )
        candidates_per_pass_slider = gr.Slider(
            0,
            64,
            value=default_config["candidates_per_pass"],
            step=1,
            label="Number of new candidates per pass. 0 means no reservoir sampling",
        )
        subsampling_factor_slider = gr.Slider(
            1,
            16,
            value=default_config["subsampling_factor"],
            step=1,
            label="Subsampling factor",
        )
        sampling_mode_dropdown = gr.Dropdown(
            choices=["gaussian", "qmc"],
            value="qmc",
            label="Sampling Mode",
            info="Select which sampling mode to use for projections",
        )

    run_button.click(
        run_sw_guidance,
        inputs=[
            num_inference_steps_slider,
            num_guided_steps_perc_slider,
            guidance_scale_slider,
            sw_u_lr_slider,
            sw_steps_slider,
            num_projections_slider,
            control_variates_dropdown,
            distance_dropdown,
            candidates_per_pass_slider,
            subsampling_factor_slider,
            sampling_mode_dropdown,
            cfg_rescale_phi_slider,
            prompt,
            reference_image,
            seed_input,
        ],
        outputs=[output_image],
    )

    clear_button = gr.ClearButton([prompt, reference_image, output_image, seed_input])

    clear_button.click(lambda: None)
