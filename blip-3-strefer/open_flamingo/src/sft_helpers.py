import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
from dataclasses import dataclass

from .helpers import exists, VisionTokenizer, PerceiverAttention, FeedForward


class InstructPerceiverResampler(VisionTokenizer):
    def __init__(
        self,
        *,
        dim_llm,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=64,
        repeat_latents=False,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        """
        Perceiver module which takes in image features and outputs image tokens.
        Args:
            dim (int): dimension of the incoming image features
            dim_inner (int, optional): final dimension to project the incoming image features to;
                also the final dimension of the outputted features. If None, no projection is used, and dim_inner = dim.
            depth (int, optional): number of layers. Defaults to 6.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            heads (int, optional): number of heads. Defaults to 8.
            num_latents (int, optional): number of latent tokens to use in the Perceiver;
                also corresponds to number of tokens per sequence to output. Defaults to 64.
            max_num_media (int, optional): maximum number of media per sequence to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            max_num_frames (int, optional): maximum number of frames to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            ff_mult (int, optional): dimension multiplier for the feedforward network. Defaults to 4.
        """
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)
        self.projection = projection

        # Text embedding projection.
        modules = [nn.Linear(dim_llm, dim)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(dim, dim))
        self.text_projection = nn.Sequential(*modules)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.repeat_latents = repeat_latents
        
        # positional embeddings
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim, dim_head=dim_head, heads=heads
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, vision_attn_masks, text_embeds=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        if self.repeat_latents:
            r = v // 729 # Repeat the query tokens for r times.
            latents = repeat(self.latents, "n d -> (n repeat) d", repeat=r)
        else:
            latents = self.latents
        latents = repeat(latents, "n d -> b T n d", b=b, T=T)
        
        if exists(text_embeds):
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds[:, None, :, :]
            latents = torch.cat((latents, text_embeds), dim=2)

        for attn, ff in self.layers:
            latents = attn(x, latents, vision_attn_masks) + latents
            latents = ff(latents) + latents
        
        # Truncate latents to only keep query tokens.
        if exists(text_embeds):
            latents = latents[:, :, :self.latents.shape[0], :]
        
        if exists(self.projection):
            return self.projection(self.norm(latents)) 
        else:
            return self.norm(latents)
        