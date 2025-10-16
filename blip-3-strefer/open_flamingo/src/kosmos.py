import torch
from einops import rearrange
from torch import nn
from typing import List, Optional, Tuple, Union

import os
import numpy

from einops import rearrange, repeat

from .helpers import PerceiverResampler
from .region_layers import MaskExtractor
from .vlm import VLMWithLanguageStream
from .sft_helpers import InstructPerceiverResampler



class Kosmos(VLMWithLanguageStream):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        image_aspect_ratio: str = 'pad',
        num_vision_tokens: int = 64,
        anyres_patch_sampling: bool = False, 
        repeat_latents: bool = False,
        max_num_vtok: int = None,
        temporal_encoder: str = 'raw',
        num_timestamp_tokens: int = None,
        mask_referring=False,
        video_mode: bool = True,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "media_token": "<image>", # 32012
            "image_placeholder_token": "<image placeholder>",
            "end_of_trunk_token": "<|endofchunk|>",
        }
        for i in range(num_timestamp_tokens):
            self._special_tokens["time_{}_token".format(i)] = "<{}>".format(i)

        if mask_referring:
            self._special_tokens["region_token"] = "<region>"
        
        lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=PerceiverResampler(
                dim=vis_feature_dim, dim_inner=lang_embedding_dim,
                num_latents=num_vision_tokens,
                repeat_latents= repeat_latents,
                max_num_vtok=max_num_vtok,
                temporal_encoder=temporal_encoder,
            ),
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            gradient_checkpointing=gradient_checkpointing,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
        )
        self.image_aspect_ratio = image_aspect_ratio
        self.anyres_patch_sampling = anyres_patch_sampling

        if mask_referring:
            self.region_encoder = MaskExtractor(
            )
        else:
            self.region_encoder = None

        self.video_mode = video_mode

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)

    def _should_apply_weight_decay(self, parameter_name):
        """
        Kosmos applies 0.01 weight deacy to everything
        """
        return True
    
    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple] = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        mask_frames=None,
        frame_nums=None,
        masks=None,
        ann_indices=None,
        **kwargs,
    ):
        """
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W) with F=1
                only F = 1 is supported (single-frame videos)
                if T_img > the number of media tokens in the corresponding input_ids (lang_x),
                only the first number of media tokens in lang_x are used
            lang_x: Language input ids, with media tokens denoting where
                visual media should be inserted.
                shape (B, T_txt)
            attention_mask: Attention mask. Defaults to None.
            labels: Labels. Defaults to None.
                shape (B, T_txt)
            past_key_values (Tuple[torch.Tensor]], optional): Past key value pairs for each of the T_txt previous tokens in the language model. Defaults to None.
                list of length = number of decoder layers in the LM
                exact implementation depends on LM, see Hugging Face docs
            past_media_locations (torch.Tensor, optional): boolean mask denoting which of the previous T_txt tokens were media tokens. Defaults to None.
                shape (B, T_txt)
            past_vision_tokens (torch.Tensor, optional): Previous vision tokens. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
                If True, includes key_values, media_locations, and vision_tokens in the output.
        """
        assert not (past_vision_tokens is None) ^ (
            past_media_locations is None
        ), "past_vision_tokens and past_media_locations must both be None or both be not None"

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            if self.image_aspect_ratio == 'anyres':
                input_dict = dict(image=vision_x, image_size=image_size)
                vision_features, vision_attn_masks = self._encode_vision_x_anyres(input_dict, lang_x.device)
            else:
                vision_features = self._encode_vision_x(vision_x=vision_x)
                vision_attn_masks = None
            
            # Same for attention masks: [b, Np, v] -> [b*Np, v]
            if self.anyres_patch_sampling:
                if self.video_mode:
                    batch_size = len(vision_x)
                    num_images = len(vision_x[0])
                    split_size = vision_features[0].shape[0]

                    vision_features = torch.stack(vision_features, dim=0)
                    vision_attn_masks = torch.stack(vision_attn_masks, dim=0)

                    vision_features = rearrange(vision_features, "(b T) p v d -> (b p) T v d", T=num_images)
                    vision_features = vision_features.unsqueeze(-3)
                    vision_attn_masks = rearrange(vision_attn_masks, "(b T) p v -> (b p) T v", T=num_images)
                    vision_attn_masks = vision_attn_masks[:, 0, :]
                else:
                    split_sizes = [feature.shape[0] for feature in vision_features]

                    # Nested splits for multi-image samples.
                    if isinstance(vision_x[0], list):
                        nt_images = [len(images) for images in vision_x]
                        split_split_sizes = []
                        img_id = 0
                        for nt in nt_images:
                            split_split_sizes.append(split_sizes[img_id:img_id+nt])
                            img_id += nt
                    else:
                        nt_images = [1] * len(vision_x)
                        split_split_sizes = split_sizes
                    vision_features = torch.cat(vision_features, dim=0)
                    vision_features = vision_features[:, None, None, :, :] # Expand dimensions.
                    vision_attn_masks = torch.cat(vision_attn_masks, dim=0)

            vision_tokens = self.vision_tokenizer(vision_features, vision_attn_masks) # <------ NOTE
            
            region_tokens = []
            region_token_nums = []
            for batch_idx in range(vision_tokens.shape[0]):
                if self.region_encoder and ann_indices and ann_indices[batch_idx]:  # mask exists
                    mask_frames_cur = [mask_frames[batch_idx]]
                    masks_cur = [masks[batch_idx]]
                    ann_indices_cur = [ann_indices[batch_idx]]
                    frame_nums_cur = [frame_nums[batch_idx]]
            
                    mask_frame_features_cur = self._encode_vision_x(
                        vision_x=torch.cat(mask_frames_cur, dim=0).unsqueeze(0)  
                        # (1, num_frames, 1, 3, H, W), e.g., torch.Size([1, 1, 1, 3, 384, 384])
                        # expect vision_x be of shape (b, T_img, F, C, H, W)
                    ).squeeze(0).squeeze(1) 
                    
                    
                    region_tokens_cur, region_token_nums_cur = self.region_encoder(
                        mask_frame_features_cur, masks_cur, 
                        vision_tokens, ann_indices_cur, frame_nums_cur)  # <------ NOTE
                    
                    region_tokens.append(region_tokens_cur)
                    region_token_nums.append(region_token_nums_cur)

                    
                else:
                    region_token_nums.append([0])

            if region_tokens:
                region_tokens = torch.cat(region_tokens, dim=0)
            
            # Post-processing: Split the batches into groups of patches and concatenate them together.
            if self.anyres_patch_sampling:
                if self.video_mode:
                    vision_tokens = rearrange(vision_tokens, "b t (T n) d -> b T (t n) d", T=num_images)
                    vision_tokens = rearrange(vision_tokens, "(b p) T n d -> b T (p n) d", p=split_size)
                else:
                    if isinstance(vision_x[0], list):
                        vision_token_groups = torch.split(vision_tokens, list(sum(nt_img) for nt_img in split_split_sizes), dim=0)
                        vision_tokens = []
                        
                        for sample_id, patch_vis_tokens in enumerate(vision_token_groups):
                            patch_vis_token_groups =  torch.split(patch_vis_tokens, split_split_sizes[sample_id], dim=0) 
                            # [Np*nt, 1, v, d] -> [[Np_t, 1, v, d], ...]
                            flatten_vision_tokens = []
                            for image_vis_token in patch_vis_token_groups:
                                image_vis_token = image_vis_token.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                                flatten_vision_tokens.append(image_vis_token)
                            vision_tokens_i = flatten_vision_tokens
                            vision_tokens.append(vision_tokens_i)
                    else:
                        vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                        vision_tokens = []
                        for patch_vis_tokens in vision_token_groups:
                            patch_vis_tokens = patch_vis_tokens.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                            vision_tokens.append(patch_vis_tokens.unsqueeze(0)) # Add the nt dimension.
        else:
            vision_tokens = None
            region_tokens = None
            region_token_nums = []

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
            region_tokens=region_tokens,
            region_token_nums=region_token_nums,
        )
        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )
        
        
        # postprocessing may be needed, e.g. to remove extra tokens from logits that were inserted into the language stream
        # or to add the past_vision_tokens and past_media_locations to the output
        output = self._postprocess_outputs_from_forward(
            output=output,
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            use_cache=use_cache,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
        )

        # postforward hooks
        self._post_forward_hook()
        return output
    
    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        image_size: Optional[Tuple] = None,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        mask_frames=None,
        frame_nums=None,
        masks=None,
        ann_indices=None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            if self.image_aspect_ratio == 'anyres':
                input_dict = dict(image=vision_x, image_size=image_size)
                vision_features, vision_attn_masks = self._encode_vision_x_anyres(input_dict, lang_x.device)
            else:
                vision_features = self._encode_vision_x(vision_x=vision_x)
                vision_attn_masks = None
            # Same for attention masks: [b, Np, v] -> [b*Np, v]
            if self.anyres_patch_sampling:
                split_sizes = [feature.shape[0] for feature in vision_features]
                # Nested splits for multi-image samples.
                if isinstance(vision_x[0], list):
                    nt_images = [len(images) for images in vision_x]
                    split_split_sizes = []
                    img_id = 0
                    for nt in nt_images:
                        split_split_sizes.append(split_sizes[img_id:img_id+nt])
                        img_id += nt
                else:
                    nt_images = [1] * len(vision_x)
                    split_split_sizes = split_sizes
                vision_features = torch.cat(vision_features, dim=0)
                vision_features = vision_features[:, None, None, :, :] # Expand dimensions.
                vision_attn_masks = torch.cat(vision_attn_masks, dim=0)

            vision_tokens = self.vision_tokenizer(vision_features, vision_attn_masks) # <------ NOTE
            
            region_tokens = []
            region_token_nums = []
            for batch_idx in range(vision_tokens.shape[0]):
                if self.region_encoder and ann_indices and ann_indices[batch_idx]:  # mask exists
                    mask_frames_cur = [mask_frames[batch_idx]]
                    masks_cur = [masks[batch_idx]]
                    ann_indices_cur = [ann_indices[batch_idx]]
                    frame_nums_cur = [frame_nums[batch_idx]]
            
            
                    mask_frame_features_cur = self._encode_vision_x(
                        vision_x=torch.cat(mask_frames_cur, dim=0).unsqueeze(0)  
                        # (1, ~bs * num_frames, 1, HxW, d_visual_encoder)
                        # expect vision_x be of shape (b, T_img, F, C, H, W)
                    ).squeeze(0).squeeze(1) 
                    
                    region_tokens_cur, region_token_nums_cur = self.region_encoder(
                        mask_frame_features_cur, masks_cur, 
                        vision_tokens, ann_indices_cur, frame_nums_cur)  # <------ NOTE
                    
                    region_tokens.append(region_tokens_cur)
                    region_token_nums.append(region_token_nums_cur)

                else:
                    region_token_nums.append([0])

            if region_tokens:
                region_tokens = torch.cat(region_tokens, dim=0)

            
            # Post-processing: Split the batches into groups of patches and concatenate them together.
            if self.anyres_patch_sampling:
                assert isinstance(vision_x, list)
                if isinstance(vision_x[0], list):
                    vision_token_groups = torch.split(vision_tokens, list(sum(nt_img) for nt_img in split_split_sizes), dim=0)
                    vision_tokens = []
                    
                    for sample_id, patch_vis_tokens in enumerate(vision_token_groups):
                        # Pad the image tokens within a sample.
                        patch_vis_token_groups =  torch.split(patch_vis_tokens, split_split_sizes[sample_id], dim=0) 
                        # [Np*nt, 1, v, d] -> [[Np_t, 1, v, d], ...]
                        # max_n_vis_token = max([vis.shape[0]*vis.shape[-2] for vis in patch_vis_token_groups])
                        flatten_vision_tokens = []
                        for image_vis_token in patch_vis_token_groups:
                            image_vis_token = image_vis_token.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                            flatten_vision_tokens.append(image_vis_token)
                            
                        vision_tokens_i = flatten_vision_tokens
                        vision_tokens.append(vision_tokens_i)
                else:
                    # Padding
                    vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                    # Padding
                    vision_tokens = []
                    for patch_vis_tokens in vision_token_groups:
                        patch_vis_tokens = patch_vis_tokens.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                        vision_tokens.append(patch_vis_tokens.unsqueeze(0)) # Add the nt dimension.
        else:
            vision_tokens = None
            region_tokens = None
            region_token_nums = []


        # fuse the vision and language tokens
        # for xattn, vision_x and media_location are repeat_interleaved s.t.
        # the total batch size is B * num_beams
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
            region_tokens=region_tokens,
            region_token_nums=region_token_nums,
        )
        if past_key_values is not None:
            output = self.lang_model.generate(
                **new_inputs,
                past_key_values=past_key_values,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        else:
            output = self.lang_model.generate(
                **new_inputs,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        self._post_forward_hook()
        return output


class KosmosInstruct(VLMWithLanguageStream):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        image_aspect_ratio: str = 'pad',
        anyres_patch_sampling: bool = False, 
        num_vision_tokens: int = 64,
        repeat_latents: bool = False,
        max_num_vtok: int = None,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "media_token": "<image>",
            "image_placeholder_token": "<image placeholder>",
            "end_of_trunk_token": "<|endofchunk|>",
        }
        lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=InstructPerceiverResampler(
                dim_llm=lang_model.get_input_embeddings().weight.shape[1], 
                dim=vis_feature_dim,
                dim_inner=lang_embedding_dim,
                num_latents=num_vision_tokens,
                repeat_latents= repeat_latents,
            ),
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            gradient_checkpointing=gradient_checkpointing,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
        )
        self.image_aspect_ratio = image_aspect_ratio
        self.anyres_patch_sampling = anyres_patch_sampling

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)

    def _should_apply_weight_decay(self, parameter_name):
        """
        Kosmos applies 0.01 weight deacy to everything
        """
        return True
    
    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple] = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W) with F=1
                only F = 1 is supported (single-frame videos)
                if T_img > the number of media tokens in the corresponding input_ids (lang_x),
                only the first number of media tokens in lang_x are used
            lang_x: Language input ids, with media tokens denoting where
                visual media should be inserted.
                shape (B, T_txt)
            attention_mask: Attention mask. Defaults to None.
            labels: Labels. Defaults to None.
                shape (B, T_txt)
            past_key_values (Tuple[torch.Tensor]], optional): Past key value pairs for each of the T_txt previous tokens in the language model. Defaults to None.
                list of length = number of decoder layers in the LM
                exact implementation depends on LM, see Hugging Face docs
            past_media_locations (torch.Tensor, optional): boolean mask denoting which of the previous T_txt tokens were media tokens. Defaults to None.
                shape (B, T_txt)
            past_vision_tokens (torch.Tensor, optional): Previous vision tokens. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
                If True, includes key_values, media_locations, and vision_tokens in the output.
        """
        assert not (past_vision_tokens is None) ^ (
            past_media_locations is None
        ), "past_vision_tokens and past_media_locations must both be None or both be not None"

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            if self.image_aspect_ratio == 'anyres':
                input_dict = dict(image=vision_x, image_size=image_size)
                vision_features, vision_attn_masks = self._encode_vision_x_anyres(input_dict, lang_x.device)
            else:
                vision_features = self._encode_vision_x(vision_x=vision_x)
                vision_attn_masks = None
            if self.anyres_patch_sampling:
                split_sizes = [feature.shape[0] for feature in vision_features]
                # Nested splits for multi-image samples.
                if isinstance(vision_x[0], list):
                    nt_images = [len(images) for images in vision_x]
                    split_split_sizes = []
                    img_id = 0
                    for nt in nt_images:
                        split_split_sizes.append(split_sizes[img_id:img_id+nt])
                        img_id += nt
                else:
                    nt_images = [1] * len(vision_x)
                    split_split_sizes = split_sizes
                vision_features = torch.cat(vision_features, dim=0)
                vision_features = vision_features[:, None, None, :, :] # Expand dimensions.
                vision_attn_masks = torch.cat(vision_attn_masks, dim=0)
            # Prepare text embeds for instruction-aware image query sampling.
            lang_x_truncated = lang_x[:, :256]
            text_embeds = self.lang_model.get_input_embeddings()(lang_x_truncated)
            # Repeat text_embeds to match the number of patches for each image patch group.
            if self.anyres_patch_sampling:
                repeated_text_embeds = []
                if text_embeds.shape[0] < len(split_sizes):
                    # Multi-image samples.
                    # text_embeds = torch.repeat_interleave(text_embeds, repeats=torch.Tensor(nt_images), dim=0)
                    text_embeds_repeat = []
                    for i, nt in enumerate(nt_images):
                        text_embeds_repeat.append(text_embeds[i].repeat(nt, 1, 1))
                    text_embeds = torch.cat(text_embeds_repeat, dim=0)
                for i, np in enumerate(split_sizes):
                    repeated_text_embeds.append(text_embeds[i].repeat(np, 1, 1))
                text_embeds = torch.cat(repeated_text_embeds, dim=0)
            vision_tokens = self.vision_tokenizer(vision_features, vision_attn_masks, text_embeds)

            # Post-processing: Split the batches into groups of patches and concatenate them together.
            if self.anyres_patch_sampling:
                if isinstance(vision_x[0], list):
                    vision_token_groups = torch.split(vision_tokens, list(sum(nt_img) for nt_img in split_split_sizes), dim=0)
                    vision_tokens = []
                    
                    for sample_id, patch_vis_tokens in enumerate(vision_token_groups):
                        # Pad the image tokens within a sample.
                        patch_vis_token_groups =  torch.split(patch_vis_tokens, split_split_sizes[sample_id], dim=0) 
                        # [Np*nt, 1, v, d] -> [[Np_t, 1, v, d], ...]
                        # max_n_vis_token = max([vis.shape[0]*vis.shape[-2] for vis in patch_vis_token_groups])
                        flatten_vision_tokens = []
                        # padded_attn_masks = []
                        for image_vis_token in patch_vis_token_groups:
                            image_vis_token = image_vis_token.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                            flatten_vision_tokens.append(image_vis_token)
                            
                        vision_tokens_i = flatten_vision_tokens
                        vision_tokens.append(vision_tokens_i)
                else:
                    # Padding
                    vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                    # Padding
                    vision_tokens = []
                    for patch_vis_tokens in vision_token_groups:
                        patch_vis_tokens = patch_vis_tokens.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                        vision_tokens.append(patch_vis_tokens.unsqueeze(0)) # Add the nt dimension.
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
        )
        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

        # postforward hooks
        self._post_forward_hook()
        return output
    
    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        image_size: Optional[Tuple] = None,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            if self.image_aspect_ratio == 'anyres':
                input_dict = dict(image=vision_x, image_size=image_size)
                vision_features, vision_attn_masks = self._encode_vision_x_anyres(input_dict, lang_x.device)
            else:
                vision_features = self._encode_vision_x(vision_x=vision_x)
                vision_attn_masks = None
            if self.anyres_patch_sampling:
                split_sizes = [feature.shape[0] for feature in vision_features]
                # Nested splits for multi-image samples.
                if isinstance(vision_x[0], list):
                    nt_images = [len(images) for images in vision_x]
                    split_split_sizes = []
                    img_id = 0
                    for nt in nt_images:
                        split_split_sizes.append(split_sizes[img_id:img_id+nt])
                        img_id += nt
                else:
                    nt_images = [1] * len(vision_x)
                    split_split_sizes = split_sizes
                vision_features = torch.cat(vision_features, dim=0)
                vision_features = vision_features[:, None, None, :, :] # Expand dimensions.
                vision_attn_masks = torch.cat(vision_attn_masks, dim=0)
            # Prepare text embeds for instruction-aware image query sampling.
            lang_x_truncated = lang_x[:, :256]
            text_embeds = self.lang_model.get_input_embeddings()(lang_x_truncated)
            # Repeat text_embeds to match the number of patches for each image patch group.
            if self.anyres_patch_sampling:
                repeated_text_embeds = []
                if text_embeds.shape[0] < len(split_sizes):
                    # Multi-image samples.
                    # text_embeds = torch.repeat_interleave(text_embeds, repeats=torch.Tensor(nt_images), dim=0)
                    text_embeds_repeat = []
                    for i, nt in enumerate(nt_images):
                        text_embeds_repeat.append(text_embeds[i].repeat(nt, 1, 1))
                    text_embeds = torch.cat(text_embeds_repeat, dim=0)
                for i, np in enumerate(split_sizes):
                    repeated_text_embeds.append(text_embeds[i].repeat(np, 1, 1))
                text_embeds = torch.cat(repeated_text_embeds, dim=0)
            vision_tokens = self.vision_tokenizer(vision_features, vision_attn_masks, text_embeds)

            # Post-processing: Split the batches into groups of patches and concatenate them together.
            if self.anyres_patch_sampling:
                assert isinstance(vision_x, list)
                if isinstance(vision_x[0], list):
                    vision_token_groups = torch.split(vision_tokens, list(sum(nt_img) for nt_img in split_split_sizes), dim=0)
                    vision_tokens = []
                    # vision_attention_mask = []
                    
                    for sample_id, patch_vis_tokens in enumerate(vision_token_groups):
                        # Pad the image tokens within a sample.
                        patch_vis_token_groups =  torch.split(patch_vis_tokens, split_split_sizes[sample_id], dim=0) 
                        # [Np*nt, 1, v, d] -> [[Np_t, 1, v, d], ...]
                        # max_n_vis_token = max([vis.shape[0]*vis.shape[-2] for vis in patch_vis_token_groups])
                        flatten_vision_tokens = []
                        # padded_attn_masks = []
                        for image_vis_token in patch_vis_token_groups:
                            image_vis_token = image_vis_token.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                            flatten_vision_tokens.append(image_vis_token)
                        vision_tokens_i = flatten_vision_tokens
                        vision_tokens.append(vision_tokens_i)
                else:
                    # Padding
                    vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                    # Padding
                    vision_tokens = []
                    for patch_vis_tokens in vision_token_groups:
                        patch_vis_tokens = patch_vis_tokens.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                        vision_tokens.append(patch_vis_tokens.unsqueeze(0)) # Add the nt dimension.
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        # for xattn, vision_x and media_location are repeat_interleaved s.t.
        # the total batch size is B * num_beams
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        if past_key_values is not None:
            output = self.lang_model.generate(
                **new_inputs,
                past_key_values=past_key_values,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        else:
            output = self.lang_model.generate(
                **new_inputs,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        self._post_forward_hook()
        return output
