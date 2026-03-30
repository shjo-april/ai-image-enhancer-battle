# Copyright 2026 Sanghyun Jo. Licensed under Apache 2.0.
# HPSv3Model: Qwen2-VL with a reward model head for visual quality scoring.
# Based on transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration,
# modified to replace the LM head with a reward prediction head (MLP).

from typing import Any

import torch
import torch.nn as nn
from transformers.cache_utils import StaticCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    QWEN2_VL_INPUTS_DOCSTRING,
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
)
from transformers.utils import add_start_docstrings_to_model_forward

_prepare_4d_causal_attention_mask_with_cache_position = (
    Qwen2VLModel._prepare_4d_causal_attention_mask_with_cache_position
)


class HPSv3Model(Qwen2VLPreTrainedModel, GenerationMixin):
    """Qwen2-VL with a reward model head for human preference scoring (HPSv3)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, output_dim: int, reward_token: str,
                 special_token_ids: list[int] | None = None,
                 rm_head_type: str = "ranknet", rm_head_kwargs: dict | None = None):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"
        self.post_init()

        # Reward model head (replaces LM head for scoring)
        self.output_dim = output_dim
        self.rm_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1024), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(1024, 16), nn.ReLU(),
            nn.Linear(16, output_dim),
        )
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate 3D RoPE position indices for vision and text tokens."""
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1],
                dtype=input_ids.dtype, device=input_ids.device,
            )
            image_index, video_index = 0, 0

            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]

                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()

                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    ed_image = input_tokens.index(image_token_id, st) if (image_token_id in input_tokens and remain_images > 0) else len(input_tokens) + 1
                    ed_video = input_tokens.index(video_token_id, st) if (video_token_id in input_tokens and remain_videos > 0) else len(input_tokens) + 1

                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t = t.item()
                    llm_grid_h = h.item() // spatial_merge_size
                    llm_grid_w = w.item() // spatial_merge_size
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)

                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(len(input_tokens) - st).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
            return position_ids, mrope_position_deltas

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False, num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs, model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens,
        )
        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas
        return model_kwargs

    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Embed tokens and merge vision embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.device, inputs_embeds.dtype))

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # LLM forward
        outputs = self.model(
            input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Reward head (in float32 for numerical stability)
        hidden_states = outputs[0]
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            logits = self.rm_head(hidden_states)

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        # Pool logits at special reward token positions
        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in self.special_token_ids:
            special_token_mask = special_token_mask | (input_ids == sid)
        pooled_logits = logits[special_token_mask, ...].view(batch_size, -1)

        return {"logits": pooled_logits}

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
        cache_position=None, position_ids=None, use_cache=True,
        pixel_values=None, pixel_values_videos=None,
        image_grid_thw=None, video_grid_thw=None, **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device).view(1, -1).expand(batch_size, -1).add(delta).unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device
            dtype = self.lm_head.weight.dtype
            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask, sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype, device=device, min_dtype=torch.finfo(dtype).min,
                cache_position=cache_position, batch_size=batch_size,
            )

        model_inputs.update({
            "position_ids": position_ids, "past_key_values": past_key_values,
            "use_cache": use_cache, "attention_mask": attention_mask,
            "pixel_values": pixel_values, "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw, "video_grid_thw": video_grid_thw,
            "rope_deltas": rope_deltas,
        })
        return model_inputs
