# Copyright 2026 Sanghyun Jo. Licensed under Apache 2.0.
# ICEdit inpainting pipeline with MoE-LoRA support.

import json
import torch
import numpy as np
import collections
import sanghyunjo as shjo
import sanghyunjo.ai_utils as shai

from typing import List

import torchvision.transforms.functional as TF

from core import diffusion
from demo_t2i_flux1 import DitPipeline


# ---------------------------------------------------------------------------
# Custom peft kwargs extractor for MoE-LoRA adapters
# (based on diffusers.utils.peft_utils.get_peft_kwargs)
# ---------------------------------------------------------------------------

def get_peft_kwargs(
    rank_dict, network_alpha_dict, peft_state_dict,
    is_unet=True, model_state_dict=None, adapter_name=None,
):
    """Extract LoRA config kwargs including MoE-specific parameters."""
    rank_pattern = {}
    alpha_pattern = {}
    r = lora_alpha = list(rank_dict.values())[0]

    # Extract MoE expert parameters
    expert_pattern = dict(filter(lambda x: "expert" in x[0], rank_dict.items()))
    expert_pattern = {k.split(".expert")[1]: v for k, v in expert_pattern.items()}
    expert_rank = collections.Counter(expert_pattern.values()).most_common()[0][0]
    expert_alpha = expert_rank
    num_experts = len(expert_pattern)

    if len(set(rank_dict.values())) > 1:
        r = collections.Counter(rank_dict.values()).most_common()[0][0]
        rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
        rank_pattern = {k.split(".lora_B.")[0]: v for k, v in rank_pattern.items()}

    if network_alpha_dict is not None and len(network_alpha_dict) > 0:
        if len(set(network_alpha_dict.values())) > 1:
            lora_alpha = collections.Counter(network_alpha_dict.values()).most_common()[0][0]
            alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
            if is_unet:
                alpha_pattern = {
                    ".".join(k.split(".lora_A.")[0].split(".")).replace(".alpha", ""): v
                    for k, v in alpha_pattern.items()
                }
            else:
                alpha_pattern = {
                    ".".join(k.split(".down.")[0].split(".")[:-1]): v
                    for k, v in alpha_pattern.items()
                }
        else:
            lora_alpha = set(network_alpha_dict.values()).pop()

    target_modules = list({name.split(".lora")[0] for name in peft_state_dict.keys()})
    use_dora = any("lora_magnitude_vector" in k for k in peft_state_dict)
    lora_bias = any("lora_B" in k and k.endswith(".bias") for k in peft_state_dict)

    return {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "target_modules": target_modules,
        "use_dora": use_dora,
        "lora_bias": lora_bias,
        "num_experts": num_experts,
        "expert_rank": expert_rank,
        "expert_alpha": expert_alpha,
    }


# ---------------------------------------------------------------------------
# ICEdit inpainting pipeline (extends DitPipeline for double-width diptych)
# ---------------------------------------------------------------------------

class DitInpaintPipeline(DitPipeline):
    def __init__(self, pipe):
        super().__init__(pipe)
        self.image_size = 512
        self.latent_size = self.image_size // self.pipe.vae_scale_factor
        self.template_prompt = (
            "A diptych with two side-by-side images of the same scene. "
            "On the right, the scene is exactly the same as on the left but"
        )
        print(f"ICEdit: Latent Size: {self.latent_size}, Image Size: {self.image_size}")

    # Override latent methods for double-width (diptych) layout

    def set_sigmas(self, steps):
        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        cfg = self.scheduler.config
        base_shift = cfg.get("base_shift", 0.5)
        max_shift = cfg.get("max_shift", 1.15)
        base_seq_len = cfg.get("base_seq_len", 256)
        max_seq_len = cfg.get("max_seq_len", 4096)
        image_seq_len = (
            (self.image_size // self.pipe.vae_scale_factor // 2)
            * (self.image_size // self.pipe.vae_scale_factor)
        )
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = image_seq_len * m + (base_shift - m * base_seq_len)
        self.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=self.device)

    def prepare_latents(self, generator=None):
        height = 2 * (self.image_size // (self.pipe.vae_scale_factor * 2))
        width = 2 * ((self.image_size * 2) // (self.pipe.vae_scale_factor * 2))
        shape = (1, self.latent_dim // 4, height, width)
        latents = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)
        packed_latents = self.pipe._pack_latents(latents, 1, self.latent_dim // 4, height, width)
        latent_ids = self.pipe._prepare_latent_image_ids(1, height // 2, width // 2, self.device, self.dtype)
        return packed_latents, latent_ids

    def unpack_latents(self, latents):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (self.image_size // (self.pipe.vae_scale_factor * 2))
        width = 2 * ((self.image_size * 2) // (self.pipe.vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // (2 * 2), height, width)

    # ==============================
    # Inpainting via ICEdit (FLUX.1-Fill-dev + MoE-LoRA)
    # ==============================
    @torch.no_grad()
    def edit(self, image: torch.Tensor, prompt: str,
             steps: int = 28, cfg: float = 50.0, generator=None) -> np.ndarray:
        # Encode text (prepend ICEdit template and force lowercase)
        prompt = self.template_prompt + " " + prompt.lower()
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_text(prompt)

        # Prepare latents and scheduler
        self.set_sigmas(steps)
        packed_latents, image_ids = self.prepare_latents(generator)
        timesteps = self.scheduler.timesteps

        # Build diptych: [original | original] with right-half mask
        combined = torch.zeros(3, self.image_size, self.image_size * 2, device=image.device, dtype=image.dtype)
        combined[:, :, :self.image_size] = image
        combined[:, :, self.image_size:] = image
        mask = torch.zeros(1, self.image_size, self.image_size * 2, device=image.device, dtype=image.dtype)
        mask[:, :, self.image_size:] = 1
        mask = mask[None].to(self.device, dtype=self.dtype)
        image = combined[None].to(self.device, dtype=self.dtype)
        masked_image = image * (1 - mask)
        masked_latents = self.encode_image(masked_image)

        # Pack mask to match latent layout
        mask = mask[:, 0, :, :]
        mask = mask.view(1, masked_latents.shape[2], self.pipe.vae_scale_factor,
                         masked_latents.shape[3], self.pipe.vae_scale_factor)
        mask = mask.permute(0, 2, 4, 1, 3)
        mask = mask.reshape(1, self.pipe.vae_scale_factor ** 2,
                            masked_latents.shape[2], masked_latents.shape[3])

        height = 2 * (self.image_size // (self.pipe.vae_scale_factor * 2))
        width = 2 * ((self.image_size * 2) // (self.pipe.vae_scale_factor * 2))

        masked_latents = self.pipe._pack_latents(masked_latents, 1, masked_latents.shape[1], height, width)
        packed_mask = self.pipe._pack_latents(mask, 1, self.pipe.vae_scale_factor ** 2, height, width)
        masked_latents = torch.cat((masked_latents, packed_mask), dim=-1)

        # Guidance
        guidance = (
            torch.ones(1, dtype=self.dtype, device=self.device) * cfg
            if self.backbone.config.guidance_embeds else None
        )

        # Denoising loop
        for t in shjo.progress(timesteps):
            noise_pred = self.backbone(
                hidden_states=torch.cat((packed_latents, masked_latents), dim=2),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t.expand(packed_latents.shape[0]).to(self.dtype) / 1000,
                guidance=guidance,
                img_ids=image_ids,
                txt_ids=text_ids,
            ).sample
            packed_latents = self.scheduler.step(noise_pred, t, packed_latents).prev_sample

        # Decode right half only (the edited region)
        unpacked_latents = self.unpack_latents(packed_latents)
        unpacked_latents = unpacked_latents[:, :, :, self.latent_size:]
        return shjo.pil2cv(self.decode_image(unpacked_latents)[0])


if __name__ == "__main__":
    args = shjo.Parser(
        {
            "arch": "FLUX.1-Fill-dev",
            "lora": "",
            "image": "./images/image_001.jpg",
            "prompt": "the colors become more vibrant and saturated with a warm golden glow.",
            "seed": 0,
            "steps": 28,
            "cfg": 50.0,
        }
    )

    import time
    from datetime import datetime

    assert args.image != "", "Editing mode requires --image argument."

    vram_before_load = torch.cuda.memory_allocated() / 1024**3

    # Replace peft modules with MoE-compatible version if needed
    if "moe" in args.lora.lower():
        import os
        import sys
        local_peft_src = os.path.abspath(
            ("/mnt/nas5/" if shjo.linux() else "//192.168.100.192/Data/") + "peft_icedit/src"
        )
        for k in list(sys.modules.keys()):
            if k.startswith("peft"):
                del sys.modules[k]
        sys.path.insert(0, local_peft_src)
        import diffusers.utils.peft_utils as peft_utils
        peft_utils.get_peft_kwargs = get_peft_kwargs

    pipe = diffusion.build_pipeline(args.arch)
    if args.lora != "":
        pipe.load_lora_weights(args.lora)
    pipe = DitInpaintPipeline(pipe)

    vram_after_load = torch.cuda.memory_allocated() / 1024**3

    pil_image = shjo.imread(args.image, "pillow").convert("RGB")
    pil_image = pil_image.resize((pipe.image_size, pipe.image_size))
    image = TF.to_tensor(pil_image) * 2 - 1

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    bgr_image = pipe.edit(
        image, args.prompt,
        steps=args.steps, cfg=args.cfg,
        generator=shai.set_seed(args.seed),
    )

    latency = time.time() - t_start
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    fig_dir = shjo.makedir(f'./figures/{args.arch}_ICEdit_{datetime.now().strftime("%y%m%d_%H%M%S")}/')
    shjo.imwrite(fig_dir + "input.jpg", shjo.pil2cv(pil_image))
    shjo.imwrite(fig_dir + "editing.jpg", bgr_image)

    log = {
        "arch": args.arch,
        "task": "ICEdit",
        "lora": args.lora,
        "image": args.image,
        "prompt": args.prompt,
        "full_prompt": pipe.template_prompt + " " + args.prompt.lower(),
        "seed": args.seed,
        "steps": args.steps,
        "cfg": args.cfg,
        "image_size": pipe.image_size,
        "vram_before_load_gb": round(vram_before_load, 3),
        "vram_after_load_gb": round(vram_after_load, 3),
        "vram_model_gb": round(vram_after_load - vram_before_load, 3),
        "vram_peak_gb": round(vram_peak, 3),
        "vram_inference_gb": round(vram_peak - vram_after_load, 3),
        "latency_sec": round(latency, 3),
    }
    shjo.jswrite(fig_dir + "log.json", log)
