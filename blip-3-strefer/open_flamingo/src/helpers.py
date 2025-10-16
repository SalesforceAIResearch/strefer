import re
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
from dataclasses import dataclass
import numpy


@dataclass
class VLMOutputWithPast(CausalLMOutputWithPast):
    """
    VLMOutputWithPast is a wrapper around CausalLMOutputWithPast that adds the following attributes:
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
    """

    past_media_locations: Optional[torch.Tensor] = None
    past_vision_tokens: Optional[torch.Tensor] = None


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def MLP(dim, inner_dim=-1, out_dim=-1):
    inner_dim = dim * 2 if inner_dim < 0 else inner_dim
    out_dim = dim if out_dim < 0 else out_dim

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, out_dim, bias=False),
    )

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(numpy.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class VisionTokenizer(nn.Module):
    def __init__(self, dim_media, num_tokens_per_media):
        super().__init__()
        self.dim_media = dim_media
        self.num_tokens_per_media = num_tokens_per_media


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, *, dim, inner_dim, heads=8):
        super().__init__()
        dim_head = inner_dim // heads
        self.scale = dim_head**-0.5
        self.heads = heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n, D)
        """
        latents = self.norm(x)

        h = self.heads

        q = self.to_q(latents)
        k = self.to_k(latents)
        v = self.to_v(latents)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return out


class TokenMixerLearnerAttentionModule(nn.Module):
    def __init__(self, *, dim, num_input_tokens, num_target_tokens):
        super().__init__()

        self.mlp = MLP(dim, inner_dim=num_target_tokens * 2, out_dim=num_target_tokens)
        self.mlp_mixer = MLP(num_input_tokens, inner_dim=num_input_tokens * 2, out_dim=num_input_tokens)

        self.norm = nn.LayerNorm(dim)
        self.num_input_tokens = num_input_tokens
        self.num_target_tokens = num_target_tokens

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n, D)
        """
        x = self.norm(x)
        x = x.transpose(-1, -2)
        x = self.mlp_mixer(x)
        x = x.transpose(-1, -2)

        inputs = self.norm(x)

        attn = self.mlp(inputs)
        attn = attn.softmax(dim=-2)

        out = einsum("... n i, ... n d -> ... i d", attn, x)

        return out


class TokenLearnerAttentionModule(nn.Module):
    def __init__(self, *, dim, num_target_tokens):
        super().__init__()

        self.mlp = MLP(dim, inner_dim=num_target_tokens * 2, out_dim=num_target_tokens)

        self.norm = nn.LayerNorm(dim)
        self.num_target_tokens = num_target_tokens

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, (T,) n, D)
        """
        inputs = self.norm(x)

        attn = self.mlp(inputs)
        attn = attn.softmax(dim=-2)

        out = einsum("... n i, ... n d -> ... i d", attn, x)

        return out


class TokenLearnerAttentionMHA(nn.Module):
    def __init__(self, *, dim, num_target_tokens, depth=4):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MultiHeadSelfAttention(dim=num_target_tokens, inner_dim=num_target_tokens),
                        MLP(dim=num_target_tokens, inner_dim=num_target_tokens * 4, out_dim=num_target_tokens),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, inner_dim=num_target_tokens * 2, out_dim=num_target_tokens)

        print('Using TokenLearner (mha) as the visual tokenizer')

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n, D)
        """
        inputs = self.norm(x)

        attn = self.mlp(inputs)

        for mha, mlp in self.layers:
            attn = mha(attn) + attn
            attn = mlp(attn) + attn

        attn = attn.softmax(dim=-2)

        out = einsum("... n i, ... n d -> ... i d", attn, x)

        return out


class TokenTuringMachineUnit(nn.Module):
    def __init__(
        self,
        *,
        dim,
        process_size=64,
        memory_size=128,
        output_size=32,
        num_layers=1,
        num_heads=8,
    ):
        super().__init__()

        self.process_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.process_layers.append(
                nn.ModuleList(
                    [
                        MultiHeadSelfAttention(
                            dim=dim, inner_dim=dim, heads=num_heads
                        ),
                        FeedForward(dim=dim, mult=4),
                    ]
                )
            )
        
        self.read_layer = TokenLearnerAttentionModule(dim=dim, num_target_tokens=process_size)
        self.write_layer = TokenLearnerAttentionModule(dim=dim, num_target_tokens=memory_size)
        self.output_layer = TokenLearnerAttentionModule(dim=dim, num_target_tokens=output_size)
    
    def forward(self, memory_tokens, input_tokens):
        """
        Args:
            memory_tokens (torch.Tensor):
                shape (b, memory_size, D)
            input_tokens (torch.Tensor):
                shape (b, n, D)
        """
        all_tokens = torch.cat([memory_tokens, input_tokens], dim=1)

        latents = self.read_layer(all_tokens)

        for attn, ff in self.process_layers:
            latents = attn(latents) + latents
            latents = ff(latents) + latents

        mem_out_tokens = torch.cat([memory_tokens, latents], dim=1)
        mem_out_tokens = self.write_layer(mem_out_tokens)

        output_tokens = self.output_layer(latents)

        return (mem_out_tokens, output_tokens)


class GroupedTokenTuringMachineUnit(nn.Module):
    def __init__(
        self,
        *,
        dim,
        process_size=128,
        memory_size_per_group=4,
        num_layers=1,
        num_heads=8,
    ):
        super().__init__()

        self.process_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.process_layers.append(
                nn.ModuleList(
                    [
                        MultiHeadSelfAttention(
                            dim=dim, inner_dim=dim, heads=num_heads
                        ),
                        FeedForward(dim=dim, mult=4),
                    ]
                )
            )
        
        self.read_layer = TokenLearnerAttentionModule(dim=dim, num_target_tokens=process_size)
        self.write_layer = TokenLearnerAttentionModule(dim=dim, num_target_tokens=memory_size_per_group)
    
    def forward(self, memory_tokens, input_tokens):
        """
        Args:
            memory_tokens (torch.Tensor):
                shape (b, n, group_memory_size, D)
            input_tokens (torch.Tensor):
                shape (b, n, D)
        """
        b, n, g, D = memory_tokens.shape

        input_tokens = input_tokens.unsqueeze(2)  # (b, n, 1, D)
        all_tokens = torch.cat([memory_tokens, input_tokens], dim=2)

        latents = all_tokens.view(b*n, g+1, D)

        for attn, ff in self.process_layers:
            latents = attn(latents) + latents
            latents = ff(latents) + latents

        # mem_out_tokens = memory_tokens.view(b*n, g, D)
        latents = latents.view(b, n, g+1, D)
        mem_out_tokens = torch.cat([memory_tokens, latents], dim=2)

        mem_out_tokens = mem_out_tokens.view(b*n, -1, D)
        mem_out_tokens = self.write_layer(mem_out_tokens)
        mem_out_tokens = mem_out_tokens.view(b, n, g, D)

        return mem_out_tokens


class GroupedMixerTokenTuringMachineUnit(nn.Module):
    def __init__(
        self,
        *,
        dim,
        process_size=128,
        memory_size_per_group=4,
        num_layers=1,
        num_heads=8,
    ):
        super().__init__()

        self.process_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.process_layers.append(
                nn.ModuleList(
                    [
                        torch.nn.Conv1d(dim, dim, 4, padding='same'),
                        # FeedForward(dim=8, mult=4),
                        FeedForward(dim=dim, mult=4),
                    ]
                )
            )
        self.gelu = nn.GELU()
        self.read_layer = TokenLearnerAttentionModule(dim=dim, num_target_tokens=process_size)
        self.write_layer = TokenLearnerAttentionModule(dim=dim, num_target_tokens=memory_size_per_group)
    
    def forward(self, memory_tokens, input_tokens):
        """
        Args:
            memory_tokens (torch.Tensor):
                shape (b, n, group_memory_size, D)
            input_tokens (torch.Tensor):
                shape (b, n, D)
        """
        b, n, g, D = memory_tokens.shape

        input_tokens = input_tokens.unsqueeze(2)  # (b, n, 1, D)
        all_tokens = torch.cat([memory_tokens, input_tokens], dim=2)

        latents = all_tokens.view(b*n, g+1, D)

        for attn, ff in self.process_layers:
            residual = latents
            latents = latents.transpose(-1, -2)
            latents = attn(latents)
            latents = self.gelu(latents)
            latents = latents.transpose(-1, -2) + residual
            
            latents = ff(latents) + latents

        # mem_out_tokens = memory_tokens.view(b*n, g, D)
        latents = latents.view(b, n, g+1, D)
        mem_out_tokens = torch.cat([memory_tokens, latents], dim=2)

        mem_out_tokens = mem_out_tokens.view(b*n, -1, D)
        mem_out_tokens = self.write_layer(mem_out_tokens)
        mem_out_tokens = mem_out_tokens.view(b, n, g, D)

        return mem_out_tokens


class TemporalTransformerSelector(nn.Module):
    def __init__(
        self,
        *,
        dim,
        output_size=32,
        num_layers=1,
        num_heads=8,
    ):
        super().__init__()

        self.process_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.process_layers.append(
                nn.ModuleList(
                    [
                        MultiHeadSelfAttention(
                            dim=dim, inner_dim=dim, heads=num_heads
                        ),
                        FeedForward(dim=dim, mult=4),
                    ]
                )
            )
        self.output_size = output_size

    def forward(self, input_tokens):
        """
        Args:
            input_tokens (torch.Tensor):
                shape (b, T, n, D)
        """
        b, T, n, D = input_tokens.shape

        latents = input_tokens.view(b, T*n, D)

        for attn, ff in self.process_layers:
            latents = attn(latents) + latents
            latents = ff(latents) + latents

        latents = latents[:, -self.output_size:, :]

        return latents.view(b, 1, -1, D)


class TokenTuringMachine(nn.Module):
    def __init__(
        self,
        *,
        dim,
        process_size=64,
        memory_size=128,
        output_size=32,
        num_layers=2,
        num_heads=8,
        final_output_only=False,
        memory_out_mode=False,
    ):
        super().__init__()

        self.ttm_unit = TokenTuringMachineUnit(
            dim=dim,
            process_size=process_size,
            memory_size=memory_size,
            output_size=output_size,
            num_layers=num_layers,
            num_heads=num_heads)

        self.initial_memory = nn.Parameter(torch.randn(memory_size, dim))

        self.final_output_only = final_output_only

        self.memory_out_mode = memory_out_mode
        if self.memory_out_mode:
            self.pos_emb = PositionalEncoding1D(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                shape (b, T, n, D)
        """
        b, T, n, D = x.shape

        output_tokens_list = []

        memory_tokens = repeat(self.initial_memory, "n d -> b n d", b=b)

        if self.memory_out_mode:
            positional_embeddings = self.pos_emb(x[:, :, 0, :])

        for i in range(T):
            step_tokens = x[:, i, :, :]

            if self.memory_out_mode:
                pos = positional_embeddings[:, i, :]
                pos = pos.unsqueeze(1)
                step_tokens = step_tokens + pos

            memory_tokens, output_tokens = self.ttm_unit(memory_tokens, step_tokens)
            output_tokens_list.append(output_tokens)

        if self.final_output_only:
            return output_tokens.unsqueeze(1)
        elif self.memory_out_mode:
            return memory_tokens.unsqueeze(1)
        else:
            output_tokens = torch.stack(output_tokens_list, dim=1)
            output_tokens = output_tokens.view(b, 1, -1, D)
            return output_tokens


class GroupedTokenTuringMachine(nn.Module):
    def __init__(
        self,
        *,
        dim,
        process_size=128,
        memory_size_per_group=4,
        num_layers=4,
        num_heads=8,
    ):
        super().__init__()

        self.ttm_unit = GroupedTokenTuringMachineUnit(
            dim=dim,
            process_size=process_size,
            memory_size_per_group=memory_size_per_group,
            num_layers=num_layers,
            num_heads=num_heads)

        self.initial_memory = nn.Parameter(torch.randn(process_size, memory_size_per_group, dim))
        
        self.pos_emb = PositionalEncoding1D(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                shape (b, T, n, D)
        """
        b, T, n, D = x.shape

        memory_tokens = repeat(self.initial_memory, "n g d -> b n g d", b=b)

        mean_x = torch.mean(x, dim=-2, keepdim=False)
        positional_embeddings = self.pos_emb(mean_x)  # (b, T, d)  # torch.Size([1, 32, 1152])

        for i in range(T):
            step_tokens = x[:, i, :, :]

            pos = positional_embeddings[:, i, :]
            pos = pos.unsqueeze(1)
            step_tokens = step_tokens + pos
            memory_tokens = self.ttm_unit(memory_tokens, step_tokens)

        memory_tokens = torch.mean(memory_tokens, dim=-2, keepdim=False)
        # memory_tokens = torch.amax(memory_tokens, dim=-2, keepdim=False)

        return memory_tokens.unsqueeze(1)


class GroupedMambaLike(nn.Module):
    def __init__(
        self,
        *,
        dim,
        process_size=128,
        memory_size_per_group=4,
    ):
        super().__init__()

        self.expand_mlp = MLP(dim=1, inner_dim=memory_size_per_group*2, out_dim=memory_size_per_group)
        self.expand_mlp2 = MLP(dim=1, inner_dim=memory_size_per_group*2, out_dim=memory_size_per_group)

        self.initial_memory = nn.Parameter(torch.randn(process_size, memory_size_per_group, dim))
        
        self.pos_emb = PositionalEncoding1D(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                shape (b, T, n, D)
        """
        b, T, n, D = x.shape

        output_tokens_list = []

        memory_tokens = repeat(self.initial_memory, "n g d -> b n g d", b=b)

        mean_x = torch.mean(x, dim=-2, keepdim=False)
        positional_embeddings = self.pos_emb(mean_x)  # (b, T, d)

        for i in range(T):
            step_tokens = x[:, i, :, :]  # (b, n, D)

            pos = positional_embeddings[:, i, :]  # (b, d)
            pos = pos.unsqueeze(1)
            step_tokens = step_tokens + pos

            # print(step_tokens.shape)
            B = step_tokens.transpose(-1, -2)  # (b, D, n)
            B = B.unsqueeze(-1)  # (b, D, n, 1)

            B = self.expand_mlp(B)  # (b, D, n, g)
            B = B.permute(0, 2, 3, 1)  # (b, n, g, D)

            A = step_tokens.transpose(-1, -2)  # (b, D, n)
            A = A.unsqueeze(-1)  # (b, D, n, 1)

            A = self.expand_mlp(A)  # (b, D, n, g)
            A = A.permute(0, 2, 3, 1)  # (b, n, g, D)
            A = torch.sigmoid(A)

            memory_tokens = A * memory_tokens + B * step_tokens.unsqueeze(-2)

        output_tokens = memory_tokens.mean(2, keepdim=False)

        return output_tokens.unsqueeze(1)
    

class TokenLearner(nn.Module):
    def __init__(
        self,
        *,
        dim,
        output_size=128,
    ):
        super().__init__()

        self.final_output = TokenLearnerAttentionModule(dim=dim, num_target_tokens=output_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                shape (b, T, n, D)
        """
        b, T, n, D = x.shape

        output_tokens = x.view(b, -1, D)
        output_tokens = self.final_output(output_tokens)

        return output_tokens.unsqueeze(1)


class TokenLearnerPerFrame(nn.Module):
    def __init__(
        self,
        *,
        dim,
        output_per_frame=4,
    ):
        super().__init__()

        self.final_output = TokenLearnerAttentionModule(dim=dim, num_target_tokens=output_per_frame)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                shape (b, T, n, D)
        """
        b, T, n, D = x.shape

        output_tokens = self.final_output(x)

        output_tokens = output_tokens.view(b, -1, D)

        return output_tokens.unsqueeze(1)


class STMax(nn.Module):
    def __init__(
        self,
        *,
        dim,
        output_per_frame=32,
        num_frames=8,
    ):
        super().__init__()

        window_size = (128 // output_per_frame, num_frames)

        self.mp = nn.MaxPool2d(window_size, stride=window_size, padding=0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                shape (b, T, n, D)
        """
        b, T, n, D = x.shape

        x = x.transpose(1, -1)  # (b, D, n, T)
        x = self.mp(x)
        x = x.transpose(1, -1)  # (b, 1, n/output_per_frame, D)

        return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        if vision_attn_masks is not None:
            vision_attn_masks = torch.cat((vision_attn_masks, 
                                            torch.ones((latents.shape[0], latents.shape[-2]), dtype=latents.dtype, device=latents.device)),
                                            dim=-1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        if vision_attn_masks is not None:
            attn_bias = torch.zeros((q.size(0), 1, 1, q.size(-2), k.size(-2)), dtype=q.dtype, device=q.device)
            vision_attn_masks = repeat(vision_attn_masks, 'b n -> b 1 1 l n', l=q.size(-2))
            attn_bias.masked_fill_(vision_attn_masks.logical_not(), float("-inf"))
            sim += attn_bias

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(VisionTokenizer):
    def __init__(
        self,
        *,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=64,
        repeat_latents=False,
        max_num_media=None,
        max_num_frames=None,
        max_num_vtok=None,
        ff_mult=4,
        temporal_encoder='raw',
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
        self.vision_token_embs = (
            nn.Parameter(torch.randn(max_num_vtok, dim))
            if exists(max_num_vtok)
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

        self.temporal_perceiver_mode = False
        print(f'temporal encoder: {temporal_encoder}')
        if temporal_encoder=='raw':
            self.temporal_encoder = lambda x : rearrange(x, "b T n d -> b (T n) d").unsqueeze(1)
        elif temporal_encoder=='max':
            self.temporal_encoder = lambda x : torch.amax(x, dim=1, keepdim=True)
        elif temporal_encoder=='mean':
            self.temporal_encoder = lambda x : torch.mean(x, dim=1, keepdim=True)
        elif temporal_encoder=='ttm':
            self.temporal_encoder = TokenTuringMachine(dim=dim, output_size=16)
        elif temporal_encoder=='ttm_m128pos':
            self.temporal_encoder = TokenTuringMachine(dim=dim, memory_size=128, memory_out_mode=True)
        elif temporal_encoder=='gttm':
            self.temporal_encoder = GroupedTokenTuringMachine(dim=dim, process_size=128, memory_size_per_group=4)
        elif temporal_encoder=='tts':
            self.temporal_encoder = TemporalTransformerSelector(dim=dim, num_layers=4, output_size=128)
        elif temporal_encoder=='tl':
            self.temporal_encoder = TokenLearner(dim=dim, output_size=32)
        elif temporal_encoder=='tl_per_frame':
            self.temporal_encoder = TokenLearnerPerFrame(dim=dim, output_per_frame=4)
        elif temporal_encoder=='st_max':
            self.temporal_encoder = STMax(dim=dim, output_per_frame=32, num_frames=8)
        elif temporal_encoder=='mambalike':
            self.temporal_encoder = GroupedMambaLike(dim=dim, process_size=128, memory_size_per_group=4)
        elif temporal_encoder=='perceiver_only':
            self.temporal_encoder = lambda x : x[:, :, :32, :]
            self.temporal_perceiver_mode = True

    def forward(self, x, vision_attn_masks):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
            vision_attn_masks (torch.Tensor): attention masks for padded visiont tokens (i.e., x)
                shape (b, v)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
            (b, 1, n, D) <video>
            ->
            (b, T, n2, D)
        """
        b, T, F, v = x.shape[:4]

        # Vision token embedding for anyres encoding.
        if vision_attn_masks is not None:
            if exists(self.vision_token_embs):
                vis_embs = repeat(self.vision_token_embs[:v], "v d -> b T F v d", b=b, T=T, F=F)
                x = x + vis_embs

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

        if self.temporal_perceiver_mode:
            x = rearrange(
                x, "b T m d -> b (T m) d",
            ).unsqueeze(1)
            latents = repeat(self.latents, "n d -> b n d", b=b).unsqueeze(1)

        for attn, ff in self.layers:
            latents = attn(x, latents, vision_attn_masks) + latents
            latents = ff(latents) + latents

        latents = self.temporal_encoder(latents)
        
        if exists(self.projection):
            return self.projection(self.norm(latents)) 
        else:
            return self.norm(latents)

class LinearPatchProjection(VisionTokenizer):
    """Linear projection from patch features to image tokens."""

    def __init__(self, mm_projector_type, *, dim_visual, dim_out, num_patches):
        super().__init__(dim_media=dim_visual, num_tokens_per_media=num_patches)
        if mm_projector_type == 'linear':
            self.proj = nn.Linear(dim_visual, dim_out)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(dim_visual, dim_out)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(dim_out, dim_out))
                self.proj = nn.Sequential(*modules)
            else:
                raise ValueError(f'Unknown projector type: {mm_projector_type}')

    def forward(self, x):
        B = x.shape[0]
        x = rearrange(x, "b T F v d -> (b T) (F v) d")
        x = self.proj(x)
        return rearrange(x, "(b T) n d -> b T n d", b=B)
    
# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt_all)
                T_txt_all >= T_txt
                If T_txt_all > T_txt, then the last T_txt text_times are used
        """

        T_txt = x.shape[1]
        assert (
            T_txt <= media_locations.shape[1]
        ), "current text cannot be longer than conditioned media locations"

        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            media_time = torch.arange(T_img, device=x.device) + 1

            # at each boolean of True, increment the time counter (relative to media time)
            text_time = media_locations.cumsum(dim=-1)[:, -T_txt:]

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(
                text_without_media_mask, "b i -> b 1 i 1"
            )
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,
        media,
        media_locations=None,
    ):
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x


class QFormerWithProjection(VisionTokenizer):
    """
    Based on BLIP-2 (https://arxiv.org/pdf/2301.12597.pdf)
    In the BLIP-2 paper, Q-former is initialized with BERT-base weights,
    so dim_inner = 768, num_hidden_layers = 12, and intermediate_size = 3072
    """

    def __init__(
        self,
        dim_input,
        dim_out,
        dim_inner=768,
        num_hidden_layers=12,
        num_query_tokens=32,
    ):
        super().__init__(dim_media=dim_out, num_tokens_per_media=num_query_tokens)
        # initialize the qformer
        from transformers import Blip2QFormerModel, Blip2QFormerConfig

        self.qformer = Blip2QFormerModel(
            Blip2QFormerConfig(
                encoder_hidden_size=dim_input,
                hidden_size=dim_inner,
                num_hidden_layers=num_hidden_layers,
            )
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, dim_inner)
        )
        self.proj = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (B, T, F, v, D)
        Returns:
            shape (B, T, n, D) where n is num_query_tokens
        """
        # HF class expects three dimensional input
        B, T = x.shape[:2]
        x = rearrange(x, "b T F v d -> (b T) (F v) d")

        # get the outputs
        image_attention_mask = torch.ones(
            x.size()[:-1], dtype=torch.long, device=x.device
        )
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=image_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        query_output = query_outputs[0]
        query_output = self.proj(query_output)

        # reshape
        query_output = rearrange(query_output, "(b T) n d -> b T n d", b=B)
        return query_output


# Both DecoupledEmbedding and DecoupledLinear are taken from https://github.com/huggingface/transformers/blob/v4.32.1/src/transformers/models/idefics/modeling_idefics.py and renamed for clarity
class DecoupledEmbedding(nn.Embedding):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0,
    then it will create `num_additional_embeddings` additional parameters that are always trained. If
    `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
    """

    def __init__(
        self,
        max_original_id: int,
        num_additional_embeddings: int = 0,
        _weight: torch.Tensor = None,
        num_original_embeddings: int = None,
        embedding_dim: int = None,
        partially_freeze=True,
        device=None,
        dtype=None,
        pad_token_id=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`):
                The largest token id that should be embedded using the regular embedding (regular `weight`).
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal self.weight.shape[0]
            num_additional_embeddings (`int`):
                Number of additional tokens to initialize an Embedding matrix for (`additional_weight`).
            _weight (`torch.Tensor`, *optional*, defaults to `None`): The regular weight tensor.
                If provided, this sets the `num_original_embeddings` and `embedding_dim` parameters.
            num_original_embeddings (`int`):
                self.weight.shape[0]
            embedding_dim (`int`):
                The size of each embedding vector
            partially_freeze: (`bool`, *optional*, defaults to `True`):
                If `True`, the regular `weight` will be frozen. `additional_weight` is never frozen.
            padding_idx (`int`, *optional*):
                The padding index (needs to be less than num_embeddings)

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`,
        `max_norm` or `norm_type`. We are not supporting these.
        """
        # validate args
        if pad_token_id is not None and pad_token_id > max_original_id:
            raise ValueError(
                f"pad_token_id must be <= max_original_id. Got {pad_token_id} and {max_original_id}."
                + "If the original tokenizer does not have a pad_token_id, use pad_token_id=None."
            )
        if _weight is not None:
            assert (num_original_embeddings is None) or (
                _weight.shape[0] == num_original_embeddings
            ), f"num_original_embeddings={num_original_embeddings} but _weight.shape[0]={_weight.shape[0]}"
            assert (embedding_dim is None) or (
                _weight.shape[1] == embedding_dim
            ), f"embedding_dim={embedding_dim} but _weight.shape[1]={_weight.shape[1]}"
            num_original_embeddings = _weight.shape[0]
            embedding_dim = _weight.shape[1]
        else:
            assert (
                num_original_embeddings is not None
            ), "num_original_embeddings must be provided if _weight is not provided"
            assert (
                embedding_dim is not None
            ), "embedding_dim must be provided if _weight is not provided"

        super().__init__(
            num_embeddings=num_original_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=pad_token_id,
            _weight=_weight,
        )
        self.max_original_id = max_original_id
        self.padding_idx = pad_token_id
        self.num_additional_embeddings = num_additional_embeddings
        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(
            require_regular_grad=not partially_freeze, require_additional_grad=True
        )

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        self.additional_embedding.requires_grad_(require_additional_grad)

    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
        embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.

        """
        if self.num_additional_embeddings == 0:
            return F.embedding(input_ids, self.weight)

        # Clone so that we don't modify the original input_ids later on
        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids > self.max_original_id)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(
            input_ids_additional_vocab - self.max_original_id - 1
        )

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)

        # overwrite the records with high indices
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    def extra_repr(self) -> str:
        return "num_original_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.max_original_id + 1,
            self.num_additional_embeddings,
            self.embedding_dim,
            (not self.weight.requires_grad),
        )


class DecoupledLinear(nn.Linear):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `additional_out_features` > 0,
    then it will create `additional_out_features * in_features` additional parameters that are always trained. If
    `additional_out_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """

    def __init__(
        self,
        max_original_id: int,
        additional_out_features: int = 0,
        _weight: torch.Tensor = None,
        _bias: torch.Tensor = None,
        in_features: int = None,
        original_out_features: int = None,
        bias: bool = True,
        partially_freeze: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`): The largest token id that should be extracted from the regular weight.
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal original_out_features - 1
            _weight: torch.Tensor, *optional*, defaults to `None`. The regular weight tensor.
                If provided, this sets the `in_features` and `original_out_features` parameters.
            _bias: torch.Tensor, *optional*, defaults to `None`. The regular bias tensor.
            in_features: int. Input hidden size.
            original_out_features: int. Original out_features of the language model's get_output_embeddings() function.
            additional_out_features: int. Number of additional trainable dimensions.
            bias: bool. Whether to include a bias term.
            partially_freeze: bool, *optional*, defaults to `True`): If `True`, the regular `weight` will be frozen.
        """
        # argument validation
        if _weight is not None:
            assert (_weight.shape[0] == original_out_features) or (
                original_out_features is None
            ), f"original_out_features={original_out_features} but _weight.shape[0]={_weight.shape[0]}"
            assert (_weight.shape[1] == in_features) or (
                in_features is None
            ), f"in_features={in_features} but _weight.shape[1]={_weight.shape[1]}"
            in_features = _weight.shape[1]
            original_out_features = _weight.shape[0]
        else:
            assert (
                in_features is not None
            ), "in_features must be provided if _weight is not provided"
            assert (
                original_out_features is not None
            ), "original_out_features must be provided if _weight is not provided"

        if _bias is not None:
            assert bias is True, "bias must be True if _bias is provided"

        # initialize original linear
        super().__init__(
            in_features, 
            original_out_features,
            bias, 
            device, 
            dtype)
        
        # set weight and bias manually
        if _weight is not None:
            self.weight = nn.Parameter(_weight)
        if _bias is not None:
            self.bias = nn.Parameter(_bias)
            
        self.in_features = in_features
        self.original_out_features = original_out_features
        self.max_original_id = max_original_id

        # initialize additional linear
        self.additional_out_features = additional_out_features
        self.has_bias = bias
        if additional_out_features > 0:
            self.additional_fc = nn.Linear(
                in_features=in_features,
                out_features=additional_out_features,
                bias=self.has_bias,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(
            require_regular_grad=not partially_freeze, require_additional_grad=True
        )

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        if self.has_bias:
            self.bias.requires_grad_(require_regular_grad)
        self.additional_fc.requires_grad_(require_additional_grad)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        output = output[..., : self.max_original_id + 1]

        if self.additional_out_features > 0:
            additional_features = F.linear(
                input, self.additional_fc.weight, self.additional_fc.bias
            )
            output = torch.cat((output, additional_features), -1)
        return output

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        return "in_features={}, out_features={}, additional_out_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.max_original_id + 1,
            self.additional_out_features,
            self.bias is not None,
            (not self.weight.requires_grad or not self.bias.requires_grad),
        )
