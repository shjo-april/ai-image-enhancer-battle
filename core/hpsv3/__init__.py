# Copyright 2026 Sanghyun Jo. Licensed under Apache 2.0.
# HPSv3 visual quality scoring runner (Qwen2-VL backbone + reward head).

import huggingface_hub
import safetensors.torch
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from core.hpsv3.model import HPSv3Model

try:
    import flash_attn
except ImportError:
    flash_attn = None
    print("Flash Attention is not installed. Falling back to SDPA.")

# Model configuration
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
HPSV3_HF_DIR = "MizzenAI/HPSv3"
HPSV3_CKPT_NAME = "HPSv3.safetensors"
TORCH_DTYPE = torch.bfloat16
OUTPUT_DIM = 2  # (mu, logvar) for uncertainty-aware prediction

# Evaluation instruction template for HPSv3
INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best.

**Visual Quality:**
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

**Text Alignment:**
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""

PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

PROMPT_WITHOUT_SPECIAL_TOKEN = """
Please provide the overall ratings of this image:
"""


class HPSv3PytorchRunner:
    """Inference runner for HPSv3 visual quality scoring model."""

    def __init__(self, device) -> None:
        self.use_special_tokens = True
        self.max_pixels = 256 * 28 * 28
        self.min_pixels = 256 * 28 * 28

        # Load processor
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, padding_side="right")
        special_tokens = ["<|Reward|>"]
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = self.processor.tokenizer.convert_tokens_to_ids(special_tokens)

        # Load model with reward head
        self.model = HPSv3Model.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            attn_implementation="flash_attention_2" if flash_attn is not None else "sdpa",
            output_dim=OUTPUT_DIM,
            reward_token="special",
            special_token_ids=special_token_ids,
            use_cache=False,
        )

        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.model.config.tokenizer_padding_side = self.processor.tokenizer.padding_side
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        # Load HPSv3 fine-tuned weights
        ckpt_path = huggingface_hub.hf_hub_download(HPSV3_HF_DIR, HPSV3_CKPT_NAME, repo_type="model")
        state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(device)
        self.model.to(dtype=TORCH_DTYPE)
        self.model.rm_head.to(torch.float32)

    def predict(self, images: list[Image.Image], prompts: list[str]) -> list[list[float]]:
        """Score image-prompt pairs. Returns list of [mu, logvar] for each pair."""
        conversations = []
        for prompt, image in zip(prompts, images):
            conversations.append([{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": (
                        INSTRUCTION.format(text_prompt=prompt)
                        + (PROMPT_WITH_SPECIAL_TOKEN if self.use_special_tokens else PROMPT_WITHOUT_SPECIAL_TOKEN)
                    )},
                ],
            }])

        # Resize images to model input resolution
        resized = []
        for image in images:
            w, h = image.size
            new_h, new_w = smart_resize(h, w, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            resized.append(image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC))

        # Tokenize and run forward pass
        batch = self.processor(
            text=self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True),
            images=resized, padding=True, return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )

        with torch.no_grad():
            reward = self.model(
                return_dict=True,
                **{k: v.to(self.model.device) for k, v in batch.items()},
            )["logits"]

        return reward.tolist()
