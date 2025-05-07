import argparse

import gradio as gr
import lightning as L

if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Launch ReSWD Gradio interface")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to create a public link for the interface",
    )
    parser.add_argument(
        "--model",
        choices=["large", "medium", "turbo"],
        default="turbo",
        help="Which SD 3.5 model variant to use (default: turbo)",
    )
    args = parser.parse_args()

    from .color_matching import create_color_matching
    from .sw_guidance import create_sw_guidance

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # ReSWD

            ReSTIR‘d, not shaken. Combining Reservoir Sampling and Sliced Wasserstein
            Distance for Variance Reduction.

            ReSWD is a method for distribution matching with reduced variance.
            """
        )
        fabric = L.Fabric(devices=1, accelerator="auto", precision="16-mixed")

        with gr.Tab(f"SW Guidance (SD 3.5 {args.model.title()})"):
            model_name = f"stabilityai/stable-diffusion-3.5-{args.model}"
            if args.model == "turbo":
                model_name = "stabilityai/stable-diffusion-3.5-large-turbo"
            create_sw_guidance(fabric, model_name)
        with gr.Tab("Color Matching"):
            create_color_matching(fabric)

    demo.queue().launch(share=args.share)
