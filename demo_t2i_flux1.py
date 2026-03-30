# Copyright 2026 Sanghyun Jo. Licensed under Apache 2.0.
# FLUX text-to-image pipeline wrapper used as the base class for ICEdit.

import json
import torch
import numpy as np
import sanghyunjo as shjo
import sanghyunjo.ai_utils as shai

from PIL import Image
from typing import List

import torchvision.transforms.functional as TF

from core import diffusion
from diffusers import FluxPipeline, AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler


class DitPipeline:
    """Lightweight wrapper around a FLUX diffusion pipeline for text-to-image generation."""

    def __init__(self, pipe: FluxPipeline):
        self.pipe: FluxPipeline = pipe
        self.pipeline_name = pipe.__class__.__name__

        self.vae: AutoencoderKL = pipe.vae
        self.backbone: FluxTransformer2DModel = pipe.transformer
        self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler

        self.device = pipe.device
        self.dtype = self.backbone.dtype

        self.latent_dim = self.vae.config.latent_channels * 4
        self.latent_size = pipe.default_sample_size
        self.image_size = self.latent_size * pipe.vae_scale_factor

        print(f"- Latent Size: {self.latent_size}, Image Size: {self.image_size}")

    def encode_text(self, prompt: str):
        return self.pipe.encode_prompt(prompt, prompt, self.device)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(images.to(dtype=self.vae.dtype)).latent_dist.mean
        return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def decode_image(self, latents: torch.Tensor) -> List[Image.Image]:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]
        return self.pipe.image_processor.postprocess(images.detach(), output_type="pil")

    def set_sigmas(self, steps):
        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        cfg = self.scheduler.config
        base_shift = cfg.get("base_shift", 0.5)
        max_shift = cfg.get("max_shift", 1.15)
        base_seq_len = cfg.get("base_seq_len", 256)
        max_seq_len = cfg.get("max_seq_len", 4096)
        image_seq_len = (self.image_size // self.pipe.vae_scale_factor // 2) ** 2
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = image_seq_len * m + (base_shift - m * base_seq_len)
        self.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=self.device)

    def prepare_latents(self, generator=None):
        height = 2 * (self.image_size // (self.pipe.vae_scale_factor * 2))
        width = 2 * (self.image_size // (self.pipe.vae_scale_factor * 2))
        shape = (1, self.latent_dim // 4, height, width)
        latents = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)
        packed_latents = self.pipe._pack_latents(latents, 1, self.latent_dim // 4, height, width)
        latent_ids = self.pipe._prepare_latent_image_ids(1, height // 2, width // 2, self.device, self.dtype)
        return packed_latents, latent_ids

    def unpack_latents(self, latents):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (self.image_size // (self.pipe.vae_scale_factor * 2))
        width = 2 * (self.image_size // (self.pipe.vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // (2 * 2), height, width)

    # ==============================
    # Text-to-Image (arch: FLUX.1-dev)
    # ==============================
    @torch.no_grad()
    def generate(self, prompt: str, steps: int = 50, cfg: float = 3.5, generator=None) -> np.ndarray:
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_text(prompt)

        self.set_sigmas(steps)
        packed_latents, image_ids = self.prepare_latents(generator)

        guidance = (
            torch.ones(1, dtype=self.dtype, device=self.device) * cfg
            if self.backbone.config.guidance_embeds else None
        )

        for t in shjo.progress(self.scheduler.timesteps):
            noise_pred = self.backbone(
                hidden_states=packed_latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t.expand(packed_latents.shape[0]).to(self.dtype) / 1000,
                guidance=guidance,
                img_ids=image_ids,
                txt_ids=text_ids,
            ).sample
            packed_latents = self.scheduler.step(noise_pred, t, packed_latents).prev_sample

        unpacked_latents = self.unpack_latents(packed_latents)
        return shjo.pil2cv(self.decode_image(unpacked_latents)[0])


if __name__ == "__main__":
    args = shjo.Parser(
        {
            "arch": "FLUX.1-dev",
            "prompt": "A photo of a cat and a dog",
            "seed": 0,
            "steps": 50,
            "cfg": 3.5,
        }
    )

    import time
    from datetime import datetime

    vram_before_load = torch.cuda.memory_allocated() / 1024**3

    pipe = diffusion.build_pipeline(args.arch)
    pipe = DitPipeline(pipe)

    vram_after_load = torch.cuda.memory_allocated() / 1024**3

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    bgr_image = pipe.generate(
        args.prompt,
        steps=args.steps, cfg=args.cfg,
        generator=shai.set_seed(args.seed),
    )

    latency = time.time() - t_start
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    fig_dir = shjo.makedir(f'./figures/{args.arch}_T2I_{datetime.now().strftime("%y%m%d_%H%M%S")}/')
    shjo.imwrite(fig_dir + "image.jpg", bgr_image)

    log = {
        "arch": args.arch,
        "task": "T2I",
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "vram_before_load_gb": round(vram_before_load, 3),
        "vram_after_load_gb": round(vram_after_load, 3),
        "vram_model_gb": round(vram_after_load - vram_before_load, 3),
        "vram_peak_gb": round(vram_peak, 3),
        "vram_inference_gb": round(vram_peak - vram_after_load, 3),
        "latency_sec": round(latency, 3),
    }
    shjo.jswrite(fig_dir + "log.json", log)
