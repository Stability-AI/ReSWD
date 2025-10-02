# ReSWD: ReSTIR‘d, not shaken. Combining Reservoir Sampling and Sliced Wasserstein Distance for Variance Reduction.

<a href="https://reservoirswd.github.io/"><img src="https://img.shields.io/badge/Project%20Page-5CE1BC.svg"></a> <a href="https://arxiv.org/abs/2510.01061"><img src="https://img.shields.io/badge/Arxiv-2510.01061-B31B1B.svg"></a> <a href="https://huggingface.co/spaces/stabilityai/reswd"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>

This is the official codebase for **ReSWD**, a state-of-the-art algorithm for distribution matching with reduced variance. It has several applications (such as diffusion guidance or color matching).


## Installation

The project uses [uv](https://github.com/astral-sh/uv) for package management. Install it via `pip install uv` if not installed.

Then run

```sh
uv sync
```

## Running

We support a CLI and gradio demo. The cli can be run with:

```sh
uv run -m src
```

It will then guide you through the different modes

The gradio demo can be executed with

```sh
uv run -m src.gradio_demo
```

### Evaluation data for color matching

In [example/test_frames_with_color_patches](example/test_frames_with_color_patches) we have uploaded the test frames, including the mean swatch values from a color checker for error calculations.

## Citation

```BibTeX
@article{boss2025reswd,
  title={ReSWD: ReSTIR‘d, not shaken. Combining Reservoir Sampling and Sliced Wasserstein Distance for Variance Reduction.},
  author={Boss, Mark and Engelhardt, Andreas and Donné, Simon and Jampani, Varun},
  journal={arXiv preprint},
  year={2025}
}
```
