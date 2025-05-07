import os

import gradio as gr
import lightning as L
import numpy as np

from src.color_matcher import train
from src.loss import VectorSWDLoss
from src.utils.image import from_torch, to_torch


def create_color_matching(fabric: L.Fabric):
    """
    Creates the Gradio interface for color matching between source and target images.
    """
    gr.Markdown(
        """
        # Color Matching
        Matches the color of source images to target images using ASC CDL parameters.
        """
    )

    gr.Markdown("## Source Images")
    with gr.Row(variant="compact"):
        source_image = gr.Image(label="Source Image", height=512)
        target_image = gr.Image(label="Target Image", height=512)
    with gr.Row(variant="compact"):
        example_paths = os.path.join("example", "color_matching")
        # Find all images in the example directory
        example_images = [
            os.path.join(example_paths, f)
            for f in os.listdir(example_paths)
            if os.path.isfile(os.path.join(example_paths, f))
            and os.path.splitext(f)[-1] in [".png", ".jpg", ".jpeg"]
        ]

        gr.Examples(
            examples=example_images,
            inputs=[source_image],
        )
        gr.Examples(
            examples=example_images,
            inputs=[target_image],
        )

    gr.Markdown(
        """
        # Configuration
        Adjust the parameters for the color matching process.
        """
    )

    with gr.Accordion("Advanced Config", open=False):
        learning_rate_slider = gr.Slider(
            1e-4, 0.1, value=1e-3, step=1e-4, label="Learning rate"
        )
        match_resolution_slider = gr.Slider(
            64, 1024, value=128, step=64, label="Match resolution"
        )
        num_steps_slider = gr.Slider(
            50, 500, value=150, step=25, label="Number of optimization steps"
        )
        control_variates_dropdown = gr.Dropdown(
            choices=["None", "Lower", "Upper"],
            value="None",
            label="Control Variates",
            info="Select which control variates to use for optimization",
        )
        candidates_per_pass_slider = gr.Slider(
            0,
            64,
            value=16,
            step=1,
            label="Number of new candidates per pass.",
            info=(
                "The number of new candidates to generate per pass in the reservoir "
                "sampling."
            ),
        )
        num_projections_slider = gr.Slider(
            16, 1024, value=64, step=16, label="Number of projections"
        )
        sampling_mode_dropdown = gr.Dropdown(
            choices=["gaussian", "qmc"],
            value="gaussian",
            label="Sampling Mode",
            info="Select which sampling mode to use for projections",
        )

    run_button = gr.Button("Run Color Matching", variant="primary")

    with gr.Column(variant="compact"):
        output_image = gr.Image(label="Color Matched Image", height=512)
        output_cdl = gr.Textbox(label="CDL Parameters")

    def run_color_matching(
        match_resolution: int,
        num_steps: int,
        control_variates: str,
        num_projections: int,
        candidates_per_pass: int,
        sampling_mode: str,
        learning_rate: float,
        *images,
    ):
        """
        Runs the color matching process between source and target images.
        """
        # Split images into source and target pair
        source_img = images[0] / 255.0
        target_img = images[1] / 255.0

        # Convert images to tensors
        source_tensors = to_torch(source_img).float().unsqueeze(0)
        target_tensors = to_torch(target_img).float().unsqueeze(0)

        # Configure loss function
        criteria = VectorSWDLoss(
            num_proj=num_projections,
            use_ucv=control_variates == "Upper",
            use_lcv=control_variates == "Lower",
            num_new_candidates=candidates_per_pass,
            sampling_mode=sampling_mode,
        )

        # Run color matching
        source_matched, cdl, losses = train(
            fabric=fabric,
            criteria=criteria,
            source_img=source_tensors,
            target_img=target_tensors,
            num_steps=num_steps,
            lr=learning_rate,
            match_resolution=match_resolution,
        )

        return [
            (from_torch(source_matched.squeeze(0)) * 255).astype(np.uint8),
            cdl.to_cdl_xml()[0],
        ]

    run_button.click(
        run_color_matching,
        inputs=[
            match_resolution_slider,
            num_steps_slider,
            control_variates_dropdown,
            num_projections_slider,
            candidates_per_pass_slider,
            sampling_mode_dropdown,
            learning_rate_slider,
            source_image,
            target_image,
        ],
        outputs=[output_image, output_cdl],
    )

    clear_button = gr.ClearButton(
        [source_image, target_image, output_image, output_cdl]
    )

    clear_button.click(lambda: None)
