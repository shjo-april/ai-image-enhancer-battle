# Copyright 2026 Sanghyun Jo. Licensed under Apache 2.0.
# Diffusion pipeline builder with model registry.

import os
import torch
import sanghyunjo as shjo

# Set cache directories for model downloads
shjo.set_env(
    {
        "CACHE": ("/mnt/nas5/" if shjo.linux() else "//192.168.100.192/Data/") + "cache/",
        "HF_HOME": ("/mnt/nas5/" if shjo.linux() else "//192.168.100.192/Data/") + "cache/huggingface",
        "TORCH_HOME": ("/mnt/nas5/" if shjo.linux() else "//192.168.100.192/Data/") + "cache/torch",
    }
)

# Authenticate with Hugging Face (set HF_TOKEN env var or run `huggingface-cli login`)
import huggingface_hub

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    huggingface_hub.login(token=hf_token)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

model_dict = {
    # FLUX.1
    "FLUX.1-dev": {
        "pretrained_model_or_path": "black-forest-labs/FLUX.1-dev",
        "torch_dtype": torch.bfloat16,
    },
    "FLUX.1-Fill-dev": {
        "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-Fill-dev",
        "torch_dtype": torch.bfloat16,
    },
}


def build_pipeline(arch, device=torch.device("cuda"), **kwargs):
    """Build and return a diffusion pipeline for the given architecture."""
    params = model_dict[arch].copy()
    params.update({"safety_checker": None, "requires_safety_checker": False})

    # Select pipeline class based on architecture name
    if "Fill" in arch:
        from diffusers import FluxFillPipeline

        pipeline_class = FluxFillPipeline
    else:
        from diffusers import AutoPipelineForText2Image

        pipeline_class = AutoPipelineForText2Image

    pipe = pipeline_class.from_pretrained(**params)

    if not hasattr(pipe, "default_sample_size"):
        pipe.default_sample_size = 64

    return pipe.to(device)
