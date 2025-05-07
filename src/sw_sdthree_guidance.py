# Implementation of the SW Guidance method with our enhanced SWD implementation
# See: https://github.com/alobashev/sw-guidance/ for the original implementation
#
# Alexander Lobashev, Maria Larchenko, Dmitry Guskov
# Color Conditional Generation with Sliced Wasserstein Guidance
# https://arxiv.org/abs/2503.19034


import gc
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import PIL
import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    XLA_AVAILABLE,
    StableDiffusion3PipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)

from src.loss.vector_swd import VectorSWDLoss
from src.utils.color_space import rgb_to_lab
from src.utils.image import from_torch, write_img

if XLA_AVAILABLE:
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import xm


def _no_grad_noise(model, *args, **kw):
    """Forward pass with grad disabled; result is returned detached."""
    with torch.no_grad():
        return model(*args, **kw, return_dict=False)[0].detach()


# ---------------- explicit pipeline forward call
class SWStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    swd: VectorSWDLoss = None

    def setup_swd(
        self,
        num_projections: int = 64,
        use_ucv: bool = False,
        use_lcv: bool = False,
        distance: Literal["l1", "l2"] = "l1",
        num_new_candidates: int = 32,
        subsampling_factor: int = 1,
        sampling_mode: Literal["gaussian", "qmc"] = "qmc",
    ):
        self.swd = VectorSWDLoss(
            num_proj=num_projections,
            distance=distance,
            use_ucv=use_ucv,
            use_lcv=use_lcv,
            num_new_candidates=num_new_candidates,
            missing_value_method="interpolate",
            ess_alpha=-1,
            sampling_mode=sampling_mode,
        ).to(self.device)
        self.subsampling_factor = subsampling_factor

    def do_sw_guidance(
        self,
        sw_steps,
        sw_u_lr,
        latents,
        t,
        prompt_embeds,
        pooled_prompt_embeds,
        pixels_ref,
        cur_iter_step,
        write_video_animation_path: Optional[str] = None,
    ):
        if sw_steps == 0:
            return latents

        if latents.shape[0] != prompt_embeds.shape[0]:
            prompt_embeds = prompt_embeds[1].unsqueeze(0)
            pooled_prompt_embeds = pooled_prompt_embeds[1].unsqueeze(0)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0])

        pixels_ref = (
            rgb_to_lab(pixels_ref.unsqueeze(0).clamp(0, 1).permute(0, 3, 1, 2))
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        csc_scaler = torch.tensor(
            [100, 2 * 128, 2 * 128], dtype=torch.bfloat16, device=latents.device
        ).view(1, 3, 1)
        csc_bias = torch.tensor(
            [0, 0.5, 0.5], dtype=torch.bfloat16, device=latents.device
        ).view(1, 3, 1)

        u = torch.nn.Parameter(
            torch.zeros_like(latents, dtype=torch.bfloat16, device=latents.device)
        )
        optimizer = torch.optim.Adam([u], lr=sw_u_lr)

        for tt in range(sw_steps):
            optimizer.zero_grad()

            x_hat_t = latents.detach() + u
            noise_pred = _no_grad_noise(
                self.transformer,
                hidden_states=x_hat_t,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
            )

            # ------------ Compute x_0
            sigma_t = self.scheduler.sigmas[
                self.scheduler.index_for_timestep(t)
            ]  # scalar
            while sigma_t.ndim < x_hat_t.ndim:
                sigma_t = sigma_t.unsqueeze(-1)
            sigma_t = sigma_t.to(x_hat_t.dtype).to(latents.device)

            x_0 = x_hat_t - sigma_t * noise_pred

            # ------------ Compute loss
            img_unscaled = self.vae.decode(
                (x_0 / self.vae.config.scaling_factor) + self.vae.config.shift_factor,
                return_dict=False,
            )[0]
            image = (img_unscaled * 0.5 + 0.5).clamp(0, 1)
            image_matched = (
                rgb_to_lab(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
            )
            # reshape to (B, D, N)  where D=3, N = H*W
            pred_seq = image_matched.view(1, 3, -1) / csc_scaler + csc_bias
            ref_seq = pixels_ref.view(1, 3, -1) / csc_scaler + csc_bias

            # Apply subsampling if enabled
            if self.subsampling_factor > 1:
                pred_seq = pred_seq[..., :: self.subsampling_factor]
                ref_seq = ref_seq[..., :: self.subsampling_factor]

            loss = self.swd(pred=pred_seq, gt=ref_seq, step=tt)

            loss.backward()
            optimizer.step()

            if write_video_animation_path is not None:
                frame_idx = cur_iter_step * sw_steps + tt
                write_img(
                    os.path.join(write_video_animation_path, f"{frame_idx:05d}.jpg"),
                    from_torch(img_unscaled.squeeze(0)),
                )

        latents = latents.detach() + u.detach()

        gc.collect()
        torch.cuda.empty_cache()
        return latents

    def __call__(
        self,
        sw_reference: PIL.Image = None,
        sw_steps: int = 8,
        sw_u_lr: float = 0.05 * 10**3,
        num_guided_steps: int = None,
        # -----------------------------------
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        cfg_rescale_phi: float = 0.7,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        write_video_animation_path: Optional[str] = None,
    ):
        assert self.swd is not None, "SWD not initialized"

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        if (
            ip_adapter_image is not None and self.is_ip_adapter_active
        ) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {
                    "ip_adapter_image_embeds": ip_adapter_image_embeds
                }
            else:
                self._joint_attention_kwargs.update(
                    ip_adapter_image_embeds=ip_adapter_image_embeds
                )

        if sw_reference is not None:
            # Resize so the reference is maximal width or height of the output image

            target_max_size = max(height, width)
            reference_max_size = max(sw_reference.width, sw_reference.height)
            scale_factor = target_max_size / reference_max_size

            sw_reference = sw_reference.resize(
                (
                    int(sw_reference.width * scale_factor),
                    int(sw_reference.height * scale_factor),
                )
            )
            pixels_ref = (
                torch.Tensor(np.array(sw_reference).astype(np.float32) / 255)
                .permute(2, 0, 1)
                .to(device)
                .to(torch.bfloat16)
            )

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible
                # with ONNX/Core ML
                timestep = t.expand(latents.shape[0])

                # SW Guidance
                if sw_reference is not None:
                    if num_guided_steps is None or i < num_guided_steps:
                        latents = self.do_sw_guidance(
                            sw_steps,
                            sw_u_lr,
                            latents,
                            t,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            pixels_ref,
                            cur_iter_step=i,
                            write_video_animation_path=write_video_animation_path,
                        )
                        if i == num_guided_steps // 2:
                            self.swd.reset()

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )

                with torch.no_grad():
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                        should_skip_layers = (
                            True
                            if i > num_inference_steps * skip_layer_guidance_start
                            and i < num_inference_steps * skip_layer_guidance_stop
                            else False
                        )
                        if skip_guidance_layers is not None and should_skip_layers:
                            timestep = t.expand(latents.shape[0])
                            latent_model_input = latents
                            noise_pred_skip_layers = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=original_prompt_embeds,
                                pooled_projections=original_pooled_prompt_embeds,
                                joint_attention_kwargs=self.joint_attention_kwargs,
                                return_dict=False,
                                skip_layers=skip_guidance_layers,
                            )[0]
                            noise_pred = (
                                noise_pred
                                + (noise_pred_text - noise_pred_skip_layers)
                                * self._skip_layer_guidance_scale
                            )

                        # Based on Sec. 3.4 of Lin, Liu, Li, Yang -
                        # Common Diffusion Noise Schedules and Sample Steps are Flawed
                        # https://arxiv.org/abs/2305.08891
                        # While Flow matching is free of most issues, a high CFG scale
                        # can still cause over-exposure issues as discussed in the work.
                        if cfg_rescale_phi is not None and cfg_rescale_phi > 0:
                            # σ_pos and σ_cfg are per-sample (B×1×1×1) stdevs
                            sigma_pos = noise_pred_text.std(dim=(1, 2, 3), keepdim=True)
                            sigma_cfg = noise_pred.std(dim=(1, 2, 3), keepdim=True)

                            # Linear blend between the raw ratio and 1,
                            # cf. Eq. (15–16) in the paper
                            factor = torch.lerp(
                                sigma_pos / (sigma_cfg + 1e-8),  # avoid div-by-zero
                                torch.ones_like(sigma_cfg),
                                1.0 - cfg_rescale_phi,
                            )
                            noise_pred = noise_pred * factor
                        else:
                            noise_pred = noise_pred

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a
                            # pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop(
                            "prompt_embeds", prompt_embeds
                        )
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )
                        negative_pooled_prompt_embeds = callback_outputs.pop(
                            "negative_pooled_prompt_embeds",
                            negative_pooled_prompt_embeds,
                        )

                    if write_video_animation_path is not None and i >= num_guided_steps:
                        with torch.no_grad():
                            image = self.vae.decode(
                                (latents / self.vae.config.scaling_factor)
                                + self.vae.config.shift_factor,
                                return_dict=False,
                            )[0]
                            cur_frame_idx = i * sw_steps
                            write_img(
                                os.path.join(
                                    write_video_animation_path,
                                    f"{cur_frame_idx:05d}.jpg",
                                ),
                                from_torch(image.squeeze(0)),
                            )

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(
                image.detach(), output_type=output_type
            )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)


def run(
    prompt: str,
    reference_image: PIL.Image.Image,
    model_path: str,
    num_inference_steps: int = 30,
    num_guided_steps: int = 28,
    guidance_scale: float = 5.0,
    cfg_rescale_phi: float = 0.7,
    sw_u_lr: float = 3e-3,
    sw_steps: int = 8,
    height: int = 768,
    width: int = 768,
    device: str = "cuda",
    seed: Optional[int] = None,
    # Add new SW-related parameters
    num_projections: int = 64,
    use_ucv: bool = False,
    use_lcv: bool = False,
    distance: Literal["l1", "l2"] = "l1",
    num_new_candidates: int = 32,
    subsampling_factor: int = 1,
    sampling_mode: Literal["gaussian", "qmc"] = "gaussian",
    pipe: Optional[SWStableDiffusion3Pipeline] = None,
    compile: bool = False,
    video_animation_path: Optional[str] = None,
) -> PIL.Image.Image:
    """
    Generate an image using SW Guidance with a given prompt and reference image.

    Args:
        prompt (str): Text prompt to guide the generation
        reference_image (PIL.Image.Image): Reference image to guide the generation
        model_path (str): Path to the model weights
        num_inference_steps (int): Number of denoising steps
        num_guided_steps (int): Number of steps to apply SW guidance
        guidance_scale (float): Scale for classifier-free guidance
        cfg_rescale_phi (float): Rescale factor for classifier-free guidance
        sw_u_lr (float): Learning rate for SW guidance
        sw_steps (int): Number of steps to apply SW guidance
        height (int): Output image height
        width (int): Output image width
        device (str): Device to run the model on
        num_projections (int): Number of random projections for VectorSWDLoss
        use_ucv (bool): Use UCV variant of VectorSWDLoss
        use_lcv (bool): Use LCV variant of VectorSWDLoss
        distance (str): Distance metric for VectorSWDLoss ("l1" or "l2")
        refresh_projections_every_n_steps (int): How often to refresh projections
        num_new_candidates (int): Number of new candidates for the reservoir
        subsampling_factor (int): Factor to subsample points for SW computation.
            Higher values reduce memory usage but may affect quality.
        sampling_mode (str): Sampling mode for VectorSWDLoss.
        pipe (SWStableDiffusion3Pipeline): Pipeline to use for generation.
            If None, a new pipeline is created.
        compile (bool): Whether to compile the pipeline.

    Returns:
        PIL.Image.Image: Generated image
    """
    # Normalize device to torch.device for robustness
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if pipe is None:
        pipe = create_pipeline(model_path, device, compile=compile)

    pipe.setup_swd(
        num_projections=num_projections,
        use_ucv=use_ucv,
        use_lcv=use_lcv,
        distance=distance,
        num_new_candidates=num_new_candidates,
        subsampling_factor=subsampling_factor,
        sampling_mode=sampling_mode,
    )

    if seed is not None:
        print(f"Using seed: {seed}")
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        num_guided_steps=num_guided_steps,
        guidance_scale=guidance_scale,
        cfg_rescale_phi=cfg_rescale_phi,
        sw_u_lr=sw_u_lr,
        sw_steps=sw_steps,
        height=height,
        width=width,
        sw_reference=reference_image,
        generator=generator,
        write_video_animation_path=video_animation_path,
    ).images[0]

    return image


def create_pipeline(model_path, device: str = "cuda", compile: bool = False):
    pipe = SWStableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    if compile:
        pipe.transformer = torch.compile(pipe.transformer)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder)
    return pipe
