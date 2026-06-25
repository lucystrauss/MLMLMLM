
from functools import reduce

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.amp import autocast
from typing import Callable, Literal
from torch.nn.attention.flex_attention import flex_attention

try:
    from flash_attn import flash_attn_func
except ImportError as e:
    print(e)
    print('flash_attn not installed, disabling Flash Attention')
    flash_attn_kvpacked_func = None
    flash_attn_func = None

from .utils import compile

try: 
    torch._dynamo.config.cache_size_limit = 5000
    flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
except:
    flex_attention_compiled = flex_attention

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
    
# def create_cross_causal_mask(q_len, k_len, device):
#     # allow query i to attend only to keys <= i
#     mask = torch.ones((q_len, k_len), device=device, dtype=torch.bool).triu(1)
#     # invert for PyTorch: True = blocked
#     return ~mask

# def create_cross_causal_mask(q_len, k_len, device):
#     """
#     Allow query at position i to attend to keys at positions 0..i
#     Block keys at positions i+1..k_len-1
    
#     Returns: mask where True = blocked, False = allowed
#     """
#     # triu(1) creates upper triangle (excluding diagonal) = 1
#     # This is EXACTLY what we want: block future positions
#     mask = torch.ones((q_len, k_len), device=device, dtype=torch.bool).triu(1)
#     return mask  # DON'T invert!

def create_cross_causal_mask(q_len, k_len, device):
    """
    Create causal mask for cross-attention.
    
    During KV-cached generation:
    - q_len = 1 (single new query)
    - k_len = current position + 1 (all accumulated keys)
    - The query is at position k_len - 1
    
    Returns: mask where True = blocked, False = allowed
    """
    if q_len == 1:
        # Single query during generation
        # Query is at position k_len - 1
        # Should attend to all positions 0 to k_len-1 (block nothing)
        return torch.zeros((1, k_len), device=device, dtype=torch.bool)
    else:
        # Training: multiple queries
        # Standard causal mask
        mask = torch.ones((q_len, k_len), device=device, dtype=torch.bool).triu(1)
        return mask

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head
 
# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale
    
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast("cuda", enabled = False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast("cuda", enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * freqs.cos() * scale ) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim = -1)

# norms
class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=10.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta

class RunningInstanceNorm(nn.Module):
    def __init__(self, dim, momentum = 0.99, eps = 1e-4, saturate = True, trainable_gain = True):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(1,1,dim))
        self.register_buffer("running_std", torch.ones(1,1,dim))
        self.saturate = saturate
        self.eps = eps
        self.momentum = momentum
        self.dim = dim
        self.trainable_gain = trainable_gain
        if self.trainable_gain:
            self.gain = nn.Parameter(torch.ones(1))
    
    def _update_stats(self, x):
        self.running_mean = self.running_mean * self.momentum + x.detach().mean(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)
        self.running_std  = (self.running_std * self.momentum + x.detach().std(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)).clip(min = self.eps)

    def forward(self, x):
        if self.training:
            self._update_stats(x)
        x = (x - self.running_mean) / self.running_std
        if self.saturate:
            x = torch.asinh(x)
        if self.trainable_gain:
            x = x * self.gain
        return x
        
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False, force_fp32=False, eps=1e-5):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

        self.eps = eps

        self.force_fp32 = force_fp32

    def forward(self, x):
        if not self.force_fp32:
            return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)
        else:
            output = F.layer_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), bias=self.beta.float(), eps=self.eps)
            return output.to(x.dtype)

class LayerScale(nn.Module):
    def __init__(self, dim, init_val = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))
    def forward(self, x):
        return x * self.scale

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv = False,
        conv_kernel_size = 3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
        return x * self.act(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    #@compile
    def forward(self, x):
        return self.ff(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm: Literal['l2', 'ln', 'dyt', 'none'] = 'none',
        differential = False,
        feat_scale = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.differential = differential

        dim_kv = dim_context if dim_context is not None else dim
        
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads
        self.is_cross_attention = dim_context is not None

        if dim_context is not None:
            if differential:
                self.to_q = nn.Linear(dim, dim * 2, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 3, bias=False)
            else:
                self.to_q = nn.Linear(dim, dim, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            if differential:
                self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
            else:
                self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        if qk_norm not in ['l2', 'ln', 'dyt','none']:
            raise ValueError(f'qk_norm must be one of ["l2", "ln", "none"], got {qk_norm}')
            
        self.qk_norm = qk_norm

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
        elif self.qk_norm == 'dyt':
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)

        self.sdp_kwargs = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )

        self.feat_scale = feat_scale

        if self.feat_scale:
            self.lambda_dc = nn.Parameter(torch.zeros(dim))
            self.lambda_hf = nn.Parameter(torch.zeros(dim))

        self.causal = causal
        if causal:
            print('Using `causal` argument disables FlexAttention. If you want to use them together, incorporate causal masking into `flex_attention_block_mask`.')

    @compile
    def apply_qk_layernorm(self, q, k):
        q_type = q.dtype
        k_type = k.dtype
        q = self.q_norm(q).to(q_type)
        k = self.k_norm(k).to(k_type)
        return q, k

    def apply_attn(self, q, k, v,
               causal=None,
               attn_mask=None,
               flex_attention_block_mask=None,
               flex_attention_score_mod=None,
               flash_attn_sliding_window=None):

        if self.num_heads != self.kv_heads:
            heads_per_kv_head = self.num_heads // self.kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

        if flex_attention_block_mask is not None or flex_attention_score_mod is not None:
            out = flex_attention_compiled(q, k, v,
                block_mask=flex_attention_block_mask,
                score_mod=flex_attention_score_mod)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=causal
            )
        return out

    def forward(
        self,
        x,
        context = None,
        rotary_pos_emb = None,
        causal = None, 
        attn_mask = None,
        flex_attention_block_mask = None,
        flex_attention_score_mod = None,
        flash_attn_sliding_window = None,
        past_kv = None,  # <-- NEW: KV cache
        use_cache = False,  # <-- NEW: whether to return cache
        start_pos = 0,  # <-- NEW: position offset for cache
        start_pos_cross=None,   # NEW -- cross-attn RoPE offset
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        # Compute Q, K, V
        if hasattr(self, 'to_q'):
            # Cross-attention path
            if self.differential:
                q, q_diff = self.to_q(x).chunk(2, dim=-1)
                q, q_diff = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, q_diff))
                q = torch.stack([q, q_diff], dim = 1)
                k, k_diff, v = self.to_kv(kv_input).chunk(3, dim=-1)
                k, k_diff, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, k_diff, v))
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q = self.to_q(x)
                q = rearrange(q, 'b n (h d) -> b h n d', h = h)
                k, v = self.to_kv(kv_input).chunk(2, dim=-1)
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Self-attention path
            if self.differential:
                q, k, v, q_diff, k_diff = self.to_qkv(x).chunk(5, dim=-1)
                q, k, v, q_diff, k_diff  = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v, q_diff, k_diff))
                q = torch.stack([q, q_diff], dim = 1)
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Apply KV caching for self-attention only
        if use_cache and not has_context and past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # Normalize q and k for cosine sim attention
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm != "none":
            q, k = self.apply_qk_layernorm(q, k)

        # Apply rotary embeddings
        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_dtype = q.dtype
            k_dtype = k.dtype
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)
            


            # ---------------------------------------------
            # CROSS-ATTENTION RoPE OFFSETTING (DROP-IN)
            # ---------------------------------------------
            
            if has_context and use_cache and past_kv is not None:

                past_k, _ = past_kv
                prev_kv_len = past_k.shape[-2]
                current_kv_len = k.shape[-2]
                new_kv_len = current_kv_len - prev_kv_len

                # compute q_len normally
                q_len = q.shape[-2]

                # --- Q: RoPE with decoder offset ---
                q_freqs = freqs[start_pos : start_pos + q_len]
                q = apply_rotary_pos_emb(q, q_freqs)

                # --- K: RoPE only on new keys ---
                k_new = k[:, :, -new_kv_len:, :]
                # compute start_pos_cross if not provided
                if start_pos_cross is None:
                    start_pos_cross = prev_kv_len
                k_freqs = freqs[start_pos_cross : start_pos_cross + new_kv_len]
                k_new = apply_rotary_pos_emb(k_new, k_freqs)

                # stitch back
                k = torch.cat([k[:, :, :-new_kv_len, :], k_new], dim=-2)

                q = q.to(v.dtype)
                k = k.to(v.dtype)


          


            # Handle KV cache: apply RoPE with correct positions
            else:
                if use_cache and not has_context and past_kv is not None:
                    # Q is at position start_pos:start_pos+q_len
                    # K is full sequence 0:start_pos+q_len
                    q_len = q.shape[-2]
                    total_len = k.shape[-2]
                    
                    # Apply RoPE to Q at positions [start_pos:start_pos+q_len]
                    q_freqs = freqs[start_pos:start_pos+q_len]
                    q = apply_rotary_pos_emb(q, q_freqs)
                    
                    # K already has RoPE applied to past keys, only apply to new keys
                    # New keys are at positions [start_pos:start_pos+q_len]
                    new_k = k[:, :, -q_len:, :]
                    k_freqs = freqs[start_pos:start_pos+q_len]
                    new_k = apply_rotary_pos_emb(new_k, k_freqs)
                    k = torch.cat([k[:, :, :-q_len, :], new_k], dim=-2)
                else:
                    # Standard RoPE application
                    if q.shape[-2] >= k.shape[-2]:
                        ratio = q.shape[-2] / k.shape[-2]
                        q_freqs, k_freqs = freqs, ratio * freqs
                    else:
                        ratio = k.shape[-2] / q.shape[-2]
                        q_freqs, k_freqs = ratio * freqs, freqs
                    q = apply_rotary_pos_emb(q, q_freqs)
                    k = apply_rotary_pos_emb(k, k_freqs)
                
                q = q.to(v.dtype)
                k = k.to(v.dtype)
        
        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        # ✅ ADD THIS DEBUG RIGHT HERE, BEFORE apply_attn:
        '''if has_context and attn_mask is not None:
            print(f"[CROSS-ATTN DEBUG] q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
            print(f"[CROSS-ATTN DEBUG] attn_mask.shape={attn_mask.shape}")
            print(f"[CROSS-ATTN DEBUG] Mask blocks {attn_mask.sum().item()} positions out of {attn_mask.numel()}")
            # Show a sample of the mask for position 0
            if attn_mask.ndim == 2:
                print(f"[CROSS-ATTN DEBUG] Mask[0, :10] = {attn_mask[0, :10]}")
            elif attn_mask.ndim == 3:
                print(f"[CROSS-ATTN DEBUG] Mask[0, 0, :10] = {attn_mask[0, 0, :10]}")'''


        # Perform attention
        if self.differential:
            q, q_diff = q.unbind(dim = 1)
            k, k_diff = k.unbind(dim = 1)
            out = self.apply_attn(q, k, v, causal = causal, attn_mask = attn_mask, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window)
            out_diff = self.apply_attn(q_diff, k_diff, v, causal = causal, attn_mask = attn_mask, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window)
            out = out - out_diff
        else:
            out = self.apply_attn(q, k, v, causal = causal, attn_mask = attn_mask, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window)

        # Merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads
        out = self.to_out(out)

        if self.feat_scale:
            out_dc = out.mean(dim=-2, keepdim=True)
            out_hf = out - out_dc
            out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

        # Return cache if requested
        if use_cache and not has_context:
            return out, (k, v)
        
        return out

class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):     

        super().__init__()

        self.dim = dim
        
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    #@compile
    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads=64,
        cross_attend=False,
        dim_context=None,
        causal=False,
        zero_init_branch_outputs=True,
        conformer=False,
        layer_ix=-1,
        remove_norms=False,
        add_rope=False,
        layer_scale=False,
        attn_kwargs={},
        ff_kwargs={},
        norm_kwargs={},
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.dim_heads = min(dim_heads, dim)
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal
        self.layer_ix = layer_ix

        # Norms
        Norm = DynamicTanh if remove_norms else LayerNorm
        self.pre_norm = Norm(dim, **norm_kwargs)

        # Self Attention
        self.self_attn = Attention(
            dim,
            dim_heads=self.dim_heads,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs,
        )
        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

        # Cross Attention
        if cross_attend:
            self.cross_attend_norm = Norm(dim, **norm_kwargs)
            self.cross_attn = Attention(
                dim,
                dim_heads=self.dim_heads,
                dim_context=dim_context,
                causal=False,        # cross-attn never causal
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )
            self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

        # Feedforward
        self.ff_norm = Norm(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()

        # Optional Conformer
        self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None
        self.conformer_scale = LayerScale(dim) if (conformer and layer_scale) else nn.Identity()

        # Optional RoPE
        self.add_rope = add_rope
        if add_rope:
            self.rope = RotaryEmbedding(self.dim_heads // 2)

    @compile
    def forward(
        self,
        x,
        context=None,
        rotary_pos_emb=None,
        self_attention_block_mask=None,
        self_attention_score_mod=None,
        cross_attention_block_mask=None,
        cross_attention_score_mod=None,
        self_attention_flash_sliding_window=None,
        cross_attention_flash_sliding_window=None,
        past_kv_self=None,
        past_kv_cross=None,
        use_cache=False,
        start_pos=0,
        start_pos_cross=None,
        **kwargs,
    ):

        # Compute RoPE if needed
        if rotary_pos_emb is None and self.add_rope:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

        # ------------------------
        # 1. SELF-ATTENTION
        # ------------------------
        x_norm = self.pre_norm(x)

        if use_cache:
            attn_out, new_kv_self = self.self_attn(
                x_norm,
                rotary_pos_emb=rotary_pos_emb,
                flex_attention_block_mask=self_attention_block_mask,
                flex_attention_score_mod=self_attention_score_mod,
                flash_attn_sliding_window=self_attention_flash_sliding_window,
                past_kv=past_kv_self,
                use_cache=True,
                start_pos=start_pos,
            )
        else:
            attn_out = self.self_attn(
                x_norm,
                rotary_pos_emb=rotary_pos_emb,
                flex_attention_block_mask=self_attention_block_mask,
                flex_attention_score_mod=self_attention_score_mod,
                flash_attn_sliding_window=self_attention_flash_sliding_window,
            )
            new_kv_self = None

        x = x + self.self_attn_scale(attn_out)

        # ------------------------
        # 2. CROSS-ATTENTION
        # ------------------------
        new_kv_cross = None

        if context is not None and self.cross_attend:

            # causal mask for q_len x k_len
            q_len, k_len = x.shape[-2], context.shape[-2]
            cross_mask = create_cross_causal_mask(q_len, k_len, x.device)

            x_norm = self.cross_attend_norm(x)

            if use_cache:

                # Cross-attn returns ONLY the output, no KV (self-attn only returns KV)
                cross_out = self.cross_attn(
                    x_norm,
                    context=context,
                    causal=False,
                    attn_mask=cross_mask,
                    flex_attention_block_mask=cross_attention_block_mask,
                    flex_attention_score_mod=cross_attention_score_mod,
                    flash_attn_sliding_window=cross_attention_flash_sliding_window,
                    past_kv=past_kv_cross,
                    use_cache=True,
                    start_pos_cross=start_pos_cross,
                )

                x = x + self.cross_attn_scale(cross_out)

                # NEW: update KV cache manually
                # The updated KV cache is inside Attention.forward (via past_kv concatenation)
                # so we retrieve it from the call above:
                # For cross-attn, Attention.forward returns ONLY output
                # but it internally appends to past_kv_cross, so we update:
                new_kv_cross = self.cross_attn.last_kv if hasattr(self.cross_attn, "last_kv") else None

                # OR: if you maintain KV outside, simply forward past_kv_cross downstream
                new_kv_cross = past_kv_cross  # pass-through cache object

            else:
                # No-cache path
                cross_out = self.cross_attn(
                    x_norm,
                    context=context,
                    causal=False,
                    attn_mask=cross_mask,
                    flex_attention_block_mask=cross_attention_block_mask,
                    flex_attention_score_mod=cross_attention_score_mod,
                    flash_attn_sliding_window=cross_attention_flash_sliding_window,
                )
                x = x + self.cross_attn_scale(cross_out)
                new_kv_cross = None

        # ------------------------
        # 3. OPTIONAL CONFORMER
        # ------------------------
        if self.conformer is not None:
            x = x + self.conformer_scale(self.conformer(x))

        # ------------------------
        # 4. FEEDFORWARD
        # ------------------------
        x = x + self.ff_scale(self.ff(self.ff_norm(x)))

        # ------------------------
        # 5. RETURN
        # ------------------------
        if use_cache:
            # return both caches
            return x, new_kv_self, new_kv_cross

        return x
    
# class TransformerBlock(nn.Module):
#     def __init__(
#             self,
#             dim,
#             dim_heads = 64,
#             cross_attend = False,
#             dim_context = None,
#             global_cond_dim = None,
#             causal = False,
#             zero_init_branch_outputs = True,
#             conformer = False,
#             layer_ix = -1,
#             remove_norms = False,
#             add_rope = False,
#             layer_scale = False,
#             attn_kwargs = {},
#             ff_kwargs = {},
#             norm_kwargs = {}
#     ):
        
#         super().__init__()
#         self.dim = dim
#         self.dim_heads = min(dim_heads,dim)
#         self.cross_attend = cross_attend
#         self.dim_context = dim_context
#         self.causal = causal
       
#         if layer_scale and zero_init_branch_outputs:
#             print('zero_init_branch_outputs is redundant with layer_scale, setting zero_init_branch_outputs to False')
#             zero_init_branch_outputs = False
            
#         self.pre_norm = LayerNorm(dim,**norm_kwargs) if not remove_norms else DynamicTanh(dim)

#         self.add_rope = add_rope

#         self.self_attn = Attention(
#             dim,
#             dim_heads = self.dim_heads,
#             causal = causal,
#             zero_init_output=zero_init_branch_outputs,
#             **attn_kwargs
#         )

#         self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

#         self.cross_attend = cross_attend
#         if cross_attend:
#             self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else DynamicTanh(dim)
#             self.cross_attn = Attention(
#                 dim,
#                 dim_heads = self.dim_heads,
#                 dim_context=dim_context,
#                 causal = causal,
#                 zero_init_output=zero_init_branch_outputs,
#                 **attn_kwargs
#             )
#             self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        
#         self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else DynamicTanh(dim)
#         self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
#         self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()

#         self.layer_ix = layer_ix

#         self.conformer = None
#         if conformer:
#             self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs)
#             self.conformer_scale = LayerScale(dim) if layer_scale else nn.Identity()

#         self.global_cond_dim = global_cond_dim

#         if global_cond_dim is not None:
#             self.to_scale_shift_gate = nn.Parameter(torch.randn(6*dim)/dim**0.5)

#         self.rope = RotaryEmbedding(self.dim_heads // 2) if add_rope else None
        
#     @compile
#     def forward(
#         self,
#         x,
#         context = None,
#         global_cond=None,
#         rotary_pos_emb = None,
#         self_attention_block_mask = None,
#         self_attention_score_mod = None,
#         cross_attention_block_mask = None,
#         cross_attention_score_mod = None,
#         self_attention_flash_sliding_window = None,
#         cross_attention_flash_sliding_window = None,
#         past_kv_self = None,  # <-- NEW: self-attention KV cache
#         use_cache = False,  # <-- NEW: whether to return cache
#         start_pos = 0,  # <-- NEW: position offset
#         start_pos_cross=None,
#         past_kv_cross=None,        # NEW — pass cross-attn cache per layer
#         **kwargs
#     ):
#         if rotary_pos_emb is None and self.add_rope:
#             rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

#         if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
            
#             scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)

#             # self-attention with adaLN
#             residual = x
#             x = self.pre_norm(x)
#             x = x * (1 + scale_self) + shift_self
            
#             if use_cache:
#                 x, new_kv_self = self.self_attn(
#                     x, 
#                     rotary_pos_emb=rotary_pos_emb, 
#                     flex_attention_block_mask=self_attention_block_mask, 
#                     flex_attention_score_mod=self_attention_score_mod, 
#                     flash_attn_sliding_window=self_attention_flash_sliding_window,
#                     past_kv=past_kv_self,
#                     use_cache=True,
#                     start_pos=start_pos
#                 )
#             else:
#                 x = self.self_attn(
#                     x, 
#                     rotary_pos_emb=rotary_pos_emb, 
#                     flex_attention_block_mask=self_attention_block_mask, 
#                     flex_attention_score_mod=self_attention_score_mod, 
#                     flash_attn_sliding_window=self_attention_flash_sliding_window
#                 )
#                 new_kv_self = None
            
#             x = x * torch.sigmoid(1 - gate_self)
#             x = self.self_attn_scale(x)
#             x = x + residual

#             if context is not None and self.cross_attend:
#                 q_len, k_len = x.shape[-2], context.shape[-2]
#                 cross_mask = create_cross_causal_mask(q_len, k_len, x.device)

#                 x = x + self.cross_attn_scale(
#                     self.cross_attn(
#                         self.cross_attend_norm(x),
#                         context=context,
#                         causal=False,
#                         attn_mask=cross_mask,
#                         flex_attention_block_mask=cross_attention_block_mask,
#                         flex_attention_score_mod=cross_attention_score_mod,
#                         flash_attn_sliding_window=cross_attention_flash_sliding_window
#                     )
#                 )
            
#             if self.conformer is not None:
#                 x = x + self.conformer_scale(self.conformer(x))

#             # feedforward with adaLN
#             residual = x
#             x = self.ff_norm(x)
#             x = x * (1 + scale_ff) + shift_ff
#             x = self.ff(x)
#             x = x * torch.sigmoid(1 - gate_ff)
#             x = self.ff_scale(x)
#             x = x + residual

#         else:
#             if use_cache:
#                 attn_out, new_kv_self = self.self_attn(
#                     self.pre_norm(x), 
#                     rotary_pos_emb=rotary_pos_emb, 
#                     flex_attention_block_mask=self_attention_block_mask, 
#                     flex_attention_score_mod=self_attention_score_mod, 
#                     flash_attn_sliding_window=self_attention_flash_sliding_window,
#                     past_kv=past_kv_self,
#                     use_cache=True,
#                     start_pos=start_pos
#                 )
#                 x = x + self.self_attn_scale(attn_out)
#             else:
#                 x = x + self.self_attn_scale(
#                     self.self_attn(
#                         self.pre_norm(x), 
#                         rotary_pos_emb=rotary_pos_emb, 
#                         flex_attention_block_mask=self_attention_block_mask, 
#                         flex_attention_score_mod=self_attention_score_mod, 
#                         flash_attn_sliding_window=self_attention_flash_sliding_window
#                     )
#                 )
#                 new_kv_self = None

#             # if context is not None and self.cross_attend:
#             #     q_len, k_len = x.shape[-2], context.shape[-2]
#             #     cross_mask = create_cross_causal_mask(q_len, k_len, x.device)

#             #     x = x + self.cross_attn_scale(
#             #         self.cross_attn(
#             #             self.cross_attend_norm(x),
#             #             context=context,
#             #             causal=False,
#             #             attn_mask=cross_mask,
#             #             past_kv=past_kv_cross,       # NEW
#             #             use_cache=use_cache,         # NEW
#             #             start_pos_cross=start_pos_cross,   # NEW
#             #             flex_attention_block_mask=cross_attention_block_mask,
#             #             flex_attention_score_mod=cross_attention_score_mod,
#             #             flash_attn_sliding_window=cross_attention_flash_sliding_window
#             #         )
#             #     )


#             if context is not None and self.cross_attend:
#                 q_len, k_len = x.shape[-2], context.shape[-2]
#                 cross_mask = create_cross_causal_mask(q_len, k_len, x.device)

#                 # NEW: retrieve past cross-attn cache for this block
#                 if past_kv_cross is None:
#                     past_kv_cross = kwargs.get("past_kv_cross", None)

#                 if use_cache:
#                     # NEW: perform cross-attention with KV cache + RoPE offset
#                     cross_out = self.cross_attn(
#                         self.cross_attend_norm(x),
#                         context=context,
#                         causal=False,
#                         attn_mask=cross_mask,
#                         flex_attention_block_mask=cross_attention_block_mask,
#                         flex_attention_score_mod=cross_attention_score_mod,
#                         flash_attn_sliding_window=cross_attention_flash_sliding_window,
#                         past_kv=past_kv_cross,          # NEW
#                         use_cache=True,                 # NEW
#                         start_pos_cross=start_pos_cross       # NEW: use decoder's current step here
#                     )

#                     x = x + self.cross_attn_scale(cross_out)

#                     # save new cross kvs so ContinuousTransformer can collect them
#                     kwargs["new_kv_cross"] = new_kv_cross

#                 else:
#                     # fallback: training / no-cache path
#                     x = x + self.cross_attn_scale(
#                         self.cross_attn(
#                             self.cross_attend_norm(x),
#                             context=context,
#                             causal=False,
#                             attn_mask=cross_mask,
#                             flex_attention_block_mask=cross_attention_block_mask,
#                             flex_attention_score_mod=cross_attention_score_mod,
#                             flash_attn_sliding_window=cross_attention_flash_sliding_window
#                         )
#                     )

                    
#             if self.conformer is not None:
#                 x = x + self.conformer_scale(self.conformer(x))

#             x = x + self.ff_scale(self.ff(self.ff_norm(x)))
        
#         if use_cache:
#             return x, new_kv_self, kwargs.get("new_kv_cross", None)
#         return x
        
class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        final_cross_attn_ix=-1,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        num_memory_tokens=0,
        sliding_window=None,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length + self.num_memory_tokens)

        self.global_cond_embedder = None
        if global_cond_dim is not None:
            self.global_cond_embedder = nn.Sequential(
                nn.Linear(global_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim * 6)
            )

        self.final_cross_attn_ix = final_cross_attn_ix

        self.sliding_window = sliding_window

        for i in range(depth):
            should_cross_attend = cross_attend and (self.final_cross_attn_ix == -1 or i <= (self.final_cross_attn_ix))
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = should_cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )
        
    def forward(
        self,
        x,
        prepend_embeds = None,
        global_cond = None,
        return_info = False,
        use_checkpointing = True,
        exit_layer_ix = None,
        past_kvs = None,  # List of KV caches per layer: [(k, v), (k, v), ...]
        use_cache = False,  # Whether to return KV caches
        start_pos = 0,  # Absolute position offset for cached sequences
        debug_kv = False,  # Enable detailed KV cache debugging
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        if debug_kv:
            print(f"\n[TRANSFORMER ENTRY] start_pos={start_pos}, x.shape={x.shape}, use_cache={use_cache}")
            if past_kvs is not None:
                print(f"[TRANSFORMER ENTRY] past_kvs: {len(past_kvs)} layers")
                if past_kvs[0] is not None:
                    print(f"[TRANSFORMER ENTRY] Layer 0 past KV: K.shape={past_kvs[0][0].shape}, V.shape={past_kvs[0][1].shape}")

        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'
            x = torch.cat((prepend_embeds, x), dim = -2)

        if self.num_memory_tokens > 0:
            memory_tokens = self.memory_tokens.expand(batch, -1, -1)
            x = torch.cat((memory_tokens, x), dim=1)

        # Compute RoPE embeddings
        if self.rotary_pos_emb is not None:
            # For KV caching, compute RoPE for the full sequence including past
            if use_cache and past_kvs is not None and past_kvs[0] is not None:
                # Total length = past cached length + current input length
                total_len = start_pos + x.shape[1]
                if debug_kv:
                    print(f"[TRANSFORMER ROPE] With cache: total_len={total_len} (start_pos={start_pos} + x.shape[1]={x.shape[1]})")
            else:
                # No cache, just use current sequence length
                total_len = x.shape[1]
                if debug_kv:
                    print(f"[TRANSFORMER ROPE] No cache: total_len={total_len}")
            
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(total_len)
            
            if debug_kv:
                print(f"[TRANSFORMER ROPE] Generated freqs.shape={rotary_pos_emb[0].shape}")
        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        if global_cond is not None and self.global_cond_embedder is not None:
            global_cond = self.global_cond_embedder(global_cond)

        # Initialize past_kvs if not provided
        if use_cache and past_kvs is None:
            past_kvs = [None] * self.depth
            if debug_kv:
                print(f"[TRANSFORMER] Initialized empty past_kvs with {self.depth} layers")
        
        new_kvs = []


        ### cross attention kv caching
        if use_cache and ("past_kvs_cross" not in kwargs or kwargs["past_kvs_cross"] is None):
            past_kvs_cross = [None] * self.depth
        else:
            past_kvs_cross = kwargs.get("past_kvs_cross", [None] * self.depth)

        new_kvs_cross = []


        # Iterate over the transformer layers
        for layer_ix, layer in enumerate(self.layers):
            past_kv_self = past_kvs[layer_ix] if (use_cache and past_kvs is not None) else None

            past_kv_cross = past_kvs_cross[layer_ix]  if (use_cache and past_kvs_cross is not None) else None


            if use_cache:
                if past_kv_cross is None:
                    start_pos_cross = 0
                else:
                    start_pos_cross = past_kv_cross[0].shape[-2]


            if debug_kv and past_kv_self is not None:
                print(f"\n[TRANSFORMER Layer {layer_ix}] Input past_kv: K.shape={past_kv_self[0].shape}, V.shape={past_kv_self[1].shape}")

            # Note: checkpointing is incompatible with KV caching
            if use_checkpointing and not use_cache:
                x = checkpoint(
                    layer, 
                    x, 
                    rotary_pos_emb=rotary_pos_emb, 
                    global_cond=global_cond, 
                    self_attention_flash_sliding_window=self.sliding_window, 
                    **kwargs
                )
                new_kv = None
            else:
                if use_cache:
                    x, new_kv, new_kv_cross = layer(
                        x, 
                        rotary_pos_emb=rotary_pos_emb, 
                        global_cond=global_cond, 
                        self_attention_flash_sliding_window=self.sliding_window,
                        past_kv_self=past_kv_self,
                        past_kv_cross=past_kv_cross,     # NEW
                        use_cache=True,
                        start_pos=start_pos,
                        start_pos_cross=start_pos_cross,
                        **kwargs
                    )


                    # NEW: retrieve cross-kv from layer (if any)
                    new_kv_cross = kwargs.pop("new_kv_cross", None)

                    
                    if debug_kv and new_kv is not None:
                        print(f"[TRANSFORMER Layer {layer_ix}] Output new_kv: K.shape={new_kv[0].shape}, V.shape={new_kv[1].shape}")
                        expected_len = start_pos + x.shape[1]
                        actual_len = new_kv[0].shape[-2]
                        if actual_len != expected_len:
                            print(f"[TRANSFORMER Layer {layer_ix}] ⚠️ WARNING: KV length mismatch! expected={expected_len}, actual={actual_len}")
                else:
                    x = layer(
                        x, 
                        rotary_pos_emb=rotary_pos_emb, 
                        global_cond=global_cond, 
                        self_attention_flash_sliding_window=self.sliding_window, 
                        **kwargs
                    )
                    new_kv = None
                    new_kv_cross = None
            
            if use_cache:
                new_kvs.append(new_kv)
                new_kvs_cross.append(new_kv_cross)


            if return_info:
                info["hidden_states"].append(x)

            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                x = x[:, self.num_memory_tokens:, :]

                if use_cache:
                    if return_info:
                        return x, new_kvs, new_kvs_cross, info
                    return x, new_kvs, new_kvs_cross

                if return_info:
                    return x, info
                
                return x

        x = x[:, self.num_memory_tokens:, :]

        x = self.project_out(x)

        if debug_kv:
            print(f"[TRANSFORMER EXIT] x.shape={x.shape}, returning {len(new_kvs)} KV caches")

        if use_cache:
            if return_info:
                return x, new_kvs, new_kvs_cross, info
            return x, new_kvs, new_kvs_cross
        
        if return_info:
            return x, info
        
        return x



# from functools import reduce

# from einops import rearrange
# from einops.layers.torch import Rearrange
# import torch
# import torch.nn.functional as F
# from torch import nn, einsum
# from torch.amp import autocast
# from typing import Callable, Literal
# from torch.nn.attention.flex_attention import flex_attention

# try:
#     from flash_attn import flash_attn_func
# except ImportError as e:
#     print(e)
#     print('flash_attn not installed, disabling Flash Attention')
#     flash_attn_kvpacked_func = None
#     flash_attn_func = None

# from .utils import compile

# try: 
#     torch._dynamo.config.cache_size_limit = 5000
#     flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
# except:
#     flex_attention_compiled = flex_attention

# def checkpoint(function, *args, **kwargs):
#     kwargs.setdefault("use_reentrant", False)
#     return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# # Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# # License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt

# def create_causal_mask(i, j, device):
#     return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
    
# '''def create_cross_causal_mask(q_len, k_len, device):
#     # allow query i to attend only to keys <= i
#     mask = torch.ones((q_len, k_len), device=device, dtype=torch.bool).triu(1)
#     # invert for PyTorch: True = blocked
#     return ~mask'''
    
# '''def create_cross_causal_mask(q_len, k_len, device):
#     """
#     Allow query at position i to attend to keys at positions 0..i
#     Block keys at positions i+1..k_len-1
    
#     Returns: mask where True = blocked, False = allowed
#     """
#     # triu(1) creates upper triangle (excluding diagonal) = 1
#     # This is EXACTLY what we want: block future positions
#     mask = torch.ones((q_len, k_len), device=device, dtype=torch.bool).triu(1)
#     return mask  # DON'T invert!'''
    
# def create_cross_causal_mask(q_len, k_len, device):
#     """
#     Create causal mask for cross-attention.
    
#     During KV-cached generation:
#     - q_len = 1 (single new query)
#     - k_len = current position + 1 (all accumulated keys)
#     - The query is at position k_len - 1
    
#     Returns: mask where True = blocked, False = allowed
#     """
#     if q_len == 1:
#         # Single query during generation
#         # Query is at position k_len - 1
#         # Should attend to all positions 0 to k_len-1 (block nothing)
#         return torch.zeros((1, k_len), device=device, dtype=torch.bool)
#     else:
#         # Training: multiple queries
#         # Standard causal mask
#         mask = torch.ones((q_len, k_len), device=device, dtype=torch.bool).triu(1)
#         return mask

# def or_reduce(masks):
#     head, *body = masks
#     for rest in body:
#         head = head | rest
#     return head

# # positional embeddings

# class AbsolutePositionalEmbedding(nn.Module):
#     def __init__(self, dim, max_seq_len):
#         super().__init__()
#         self.scale = dim ** -0.5
#         self.max_seq_len = max_seq_len
#         self.emb = nn.Embedding(max_seq_len, dim)

#     def forward(self, x, pos = None, seq_start_pos = None):
#         seq_len, device = x.shape[1], x.device
#         assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

#         if pos is None:
#             pos = torch.arange(seq_len, device = device)

#         if seq_start_pos is not None:
#             pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

#         pos_emb = self.emb(pos)
#         pos_emb = pos_emb * self.scale
#         return pos_emb

# class ScaledSinusoidalEmbedding(nn.Module):
#     def __init__(self, dim, theta = 10000):
#         super().__init__()
#         assert (dim % 2) == 0, 'dimension must be divisible by 2'
#         self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

#         half_dim = dim // 2
#         freq_seq = torch.arange(half_dim).float() / half_dim
#         inv_freq = theta ** -freq_seq
#         self.register_buffer('inv_freq', inv_freq, persistent = False)

#     def forward(self, x, pos = None, seq_start_pos = None):
#         seq_len, device = x.shape[1], x.device

#         if pos is None:
#             pos = torch.arange(seq_len, device = device)

#         if seq_start_pos is not None:
#             pos = pos - seq_start_pos[..., None]

#         emb = einsum('i, j -> i j', pos, self.inv_freq)
#         emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
#         return emb * self.scale
    
# class RotaryEmbedding(nn.Module):
#     def __init__(
#         self,
#         dim,
#         use_xpos = False,
#         scale_base = 512,
#         interpolation_factor = 1.,
#         base = 10000,
#         base_rescale_factor = 1.
#     ):
#         super().__init__()
#         # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
#         # has some connection to NTK literature
#         # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
#         base *= base_rescale_factor ** (dim / (dim - 2))

#         inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer('inv_freq', inv_freq)

#         assert interpolation_factor >= 1.
#         self.interpolation_factor = interpolation_factor

#         if not use_xpos:
#             self.register_buffer('scale', None)
#             return

#         scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

#         self.scale_base = scale_base
#         self.register_buffer('scale', scale)

#     def forward_from_seq_len(self, seq_len):
#         device = self.inv_freq.device

#         t = torch.arange(seq_len, device = device)
#         return self.forward(t)

#     @autocast("cuda", enabled = False)
#     def forward(self, t):
#         device = self.inv_freq.device

#         t = t.to(torch.float32)

#         t = t / self.interpolation_factor

#         freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
#         freqs = torch.cat((freqs, freqs), dim = -1)

#         if self.scale is None:
#             return freqs, 1.

#         power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
#         scale = self.scale ** rearrange(power, 'n -> n 1')
#         scale = torch.cat((scale, scale), dim = -1)

#         return freqs, scale

# def rotate_half(x):
#     x = rearrange(x, '... (j d) -> ... j d', j = 2)
#     x1, x2 = x.unbind(dim = -2)
#     return torch.cat((-x2, x1), dim = -1)

# @autocast("cuda", enabled = False)
# def apply_rotary_pos_emb(t, freqs, scale = 1):
#     out_dtype = t.dtype

#     # cast to float32 if necessary for numerical stability
#     dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
#     rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
#     freqs, t = freqs.to(dtype), t.to(dtype)
#     freqs = freqs[-seq_len:, :]

#     if t.ndim == 4 and freqs.ndim == 3:
#         freqs = rearrange(freqs, 'b n d -> b 1 n d')

#     # partial rotary embeddings, Wang et al. GPT-J
#     t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

#     t = (t * freqs.cos() * scale ) + (rotate_half(t) * freqs.sin() * scale)

#     t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

#     return torch.cat((t, t_unrotated), dim = -1)

# # norms
# class DynamicTanh(nn.Module):
#     def __init__(self, dim, init_alpha=10.0):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
#         self.gamma = nn.Parameter(torch.ones(dim))
#         self.beta = nn.Parameter(torch.zeros(dim))

#     def forward(self, x):
#         x = F.tanh(self.alpha * x)
#         return self.gamma * x + self.beta

# class RunningInstanceNorm(nn.Module):
#     def __init__(self, dim, momentum = 0.99, eps = 1e-4, saturate = True, trainable_gain = True):
#         super().__init__()
#         self.register_buffer("running_mean", torch.zeros(1,1,dim))
#         self.register_buffer("running_std", torch.ones(1,1,dim))
#         self.saturate = saturate
#         self.eps = eps
#         self.momentum = momentum
#         self.dim = dim
#         self.trainable_gain = trainable_gain
#         if self.trainable_gain:
#             self.gain = nn.Parameter(torch.ones(1))
    
#     def _update_stats(self, x):
#         self.running_mean = self.running_mean * self.momentum + x.detach().mean(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)
#         self.running_std  = (self.running_std * self.momentum + x.detach().std(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)).clip(min = self.eps)

#     def forward(self, x):
#         if self.training:
#             self._update_stats(x)
#         x = (x - self.running_mean) / self.running_std
#         if self.saturate:
#             x = torch.asinh(x)
#         if self.trainable_gain:
#             x = x * self.gain
#         return x
        
# class LayerNorm(nn.Module):
#     def __init__(self, dim, bias=False, fix_scale=False, force_fp32=False, eps=1e-5):
#         """
#         bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
#         """
#         super().__init__()

#         if fix_scale:
#             self.register_buffer("gamma", torch.ones(dim))
#         else:
#             self.gamma = nn.Parameter(torch.ones(dim))

#         if bias:
#             self.beta = nn.Parameter(torch.zeros(dim))
#         else:
#             self.register_buffer("beta", torch.zeros(dim))

#         self.eps = eps

#         self.force_fp32 = force_fp32

#     def forward(self, x):
#         if not self.force_fp32:
#             return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)
#         else:
#             output = F.layer_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), bias=self.beta.float(), eps=self.eps)
#             return output.to(x.dtype)

# class LayerScale(nn.Module):
#     def __init__(self, dim, init_val = 1e-5):
#         super().__init__()
#         self.scale = nn.Parameter(torch.full([dim], init_val))
#     def forward(self, x):
#         return x * self.scale

# # feedforward

# class GLU(nn.Module):
#     def __init__(
#         self,
#         dim_in,
#         dim_out,
#         activation: Callable,
#         use_conv = False,
#         conv_kernel_size = 3,
#     ):
#         super().__init__()
#         self.act = activation
#         self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
#         self.use_conv = use_conv

#     def forward(self, x):
#         if self.use_conv:
#             x = rearrange(x, 'b n d -> b d n')
#             x = self.proj(x)
#             x = rearrange(x, 'b d n -> b n d')
#         else:
#             x = self.proj(x)

#         x, gate = x.chunk(2, dim = -1)
#         return x * self.act(gate)

# class FeedForward(nn.Module):
#     def __init__(
#         self,
#         dim,
#         dim_out = None,
#         mult = 4,
#         no_bias = False,
#         glu = True,
#         use_conv = False,
#         conv_kernel_size = 3,
#         zero_init_output = True,
#     ):
#         super().__init__()
#         inner_dim = int(dim * mult)

#         # Default to SwiGLU

#         activation = nn.SiLU()

#         dim_out = dim if dim_out is None else dim_out

#         if glu:
#             linear_in = GLU(dim, inner_dim, activation)
#         else:
#             linear_in = nn.Sequential(
#                 Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
#                 nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
#                 Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
#                 activation
#             )

#         linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

#         # init last linear layer to 0
#         if zero_init_output:
#             nn.init.zeros_(linear_out.weight)
#             if not no_bias:
#                 nn.init.zeros_(linear_out.bias)


#         self.ff = nn.Sequential(
#             linear_in,
#             Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
#             linear_out,
#             Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
#         )

#     #@compile
#     def forward(self, x):
#         return self.ff(x)

# class Attention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         dim_heads = 64,
#         dim_context = None,
#         causal = False,
#         zero_init_output=True,
#         qk_norm: Literal['l2', 'ln', 'dyt', 'none'] = 'none',
#         differential = False,
#         feat_scale = False
#     ):
#         super().__init__()
#         self.dim = dim
#         self.dim_heads = dim_heads

#         self.differential = differential

#         dim_kv = dim_context if dim_context is not None else dim
        
#         self.num_heads = dim // dim_heads
#         self.kv_heads = dim_kv // dim_heads

#         if dim_context is not None:
#             if differential:
#                 self.to_q = nn.Linear(dim, dim * 2, bias=False)
#                 self.to_kv = nn.Linear(dim_kv, dim_kv * 3, bias=False)
#             else:
#                 self.to_q = nn.Linear(dim, dim, bias=False)
#                 self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
#         else:
#             if differential:
#                 self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
#             else:
#                 self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

#         self.to_out = nn.Linear(dim, dim, bias=False)

#         if zero_init_output:
#             nn.init.zeros_(self.to_out.weight)

#         if qk_norm not in ['l2', 'ln', 'dyt','none']:
#             raise ValueError(f'qk_norm must be one of ["l2", "ln", "none"], got {qk_norm}')
            
#         self.qk_norm = qk_norm

#         if self.qk_norm == "ln":
#             self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
#             self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
#         elif self.qk_norm == 'dyt':
#             self.q_norm = DynamicTanh(dim_heads)
#             self.k_norm = DynamicTanh(dim_heads)

#         self.sdp_kwargs = dict(
#             enable_flash = True,
#             enable_math = True,
#             enable_mem_efficient = True
#         )

#         self.feat_scale = feat_scale

#         if self.feat_scale:
#             self.lambda_dc = nn.Parameter(torch.zeros(dim))
#             self.lambda_hf = nn.Parameter(torch.zeros(dim))

#         self.causal = causal
#         if causal:
#             print('Using `causal` argument disables FlexAttention. If you want to use them together, incorporate causal masking into `flex_attention_block_mask`.')

#     @compile
#     def apply_qk_layernorm(self, q, k):
#         q_type = q.dtype
#         k_type = k.dtype
#         q = self.q_norm(q).to(q_type)
#         k = self.k_norm(k).to(k_type)
#         return q, k


    # '''def apply_attn(self, q, k, v, causal = None, flex_attention_block_mask = None, flex_attention_score_mod = None, flash_attn_sliding_window = None):

    #     if self.num_heads != self.kv_heads:
    #          # Repeat interleave kv_heads to match q_heads for grouped query attention
    #          heads_per_kv_head = self.num_heads // self.kv_heads
    #          k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

    #     flash_attn_available = flash_attn_func is not None
    #     if flash_attn_sliding_window is not None and (not flash_attn_available):
    #         print(f"Cannot use FlashAttention sliding window as FlashAttention is disabled or not available")

    #     if (flex_attention_block_mask is not None or flex_attention_score_mod is not None) and flash_attn_sliding_window is not None:
    #         print(f"cannot use both FlashAttention and FlexAttention, favouring FlexAttention")

    #     if causal and (flex_attention_block_mask is not None or flex_attention_score_mod is not None):
    #         print(f"Disabling FlexAttention because causal is set")
    #         flex_attention_block_mask = None
    #         flex_attention_score_mod = None

    #     if flex_attention_block_mask is not None or flex_attention_score_mod is not None:
    #         out = flex_attention_compiled(q,k,v,
    #             block_mask = flex_attention_block_mask,
    #             score_mod = flex_attention_score_mod)        
    #     elif flash_attn_available:
    #         fa_dtype_in = q.dtype
    #         q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k, v))

    #         if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
    #             q, k, v = map(lambda t: t.to(torch.float16), (q, k, v))
            
    #         out = flash_attn_func(q, k, v, causal = causal, window_size=flash_attn_sliding_window if (flash_attn_sliding_window is not None) else [-1,-1])
            
    #         out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
    #     else:
    #         out = F.scaled_dot_product_attention(q, k, v, is_causal = causal)
    #     return out'''
        
        
#     # with cross-attn causal mask (can't just use same mask as self-attn in case cross-attn tokens are different length)
#     def apply_attn(self, q, k, v,
#                causal=None,
#                attn_mask=None,
#                flex_attention_block_mask=None,
#                flex_attention_score_mod=None,
#                flash_attn_sliding_window=None):

#         if self.num_heads != self.kv_heads:
#             heads_per_kv_head = self.num_heads // self.kv_heads
#             k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

#         if flex_attention_block_mask is not None or flex_attention_score_mod is not None:
#             out = flex_attention_compiled(q, k, v,
#                 block_mask=flex_attention_block_mask,
#                 score_mod=flex_attention_score_mod)
#         else:
#             out = F.scaled_dot_product_attention(
#                 q, k, v,
#                 attn_mask=attn_mask,
#                 is_causal=causal
#             )
#         return out



#     #@compile
#     def forward(
#         self,
#         x,
#         context = None,
#         rotary_pos_emb = None,
#         causal = None, 
#         attn_mask = None,
#         flex_attention_block_mask = None,
#         flex_attention_score_mod = None,
#         flash_attn_sliding_window = None
#     ):
#         h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

#         kv_input = context if has_context else x

#         if hasattr(self, 'to_q'):
#             # Use separate linear projections for q and k/v
#             if self.differential:
#                 q, q_diff = self.to_q(x).chunk(2, dim=-1)
#                 q, q_diff = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, q_diff))
#                 q = torch.stack([q, q_diff], dim = 1)
#                 k, k_diff, v = self.to_kv(kv_input).chunk(3, dim=-1)
#                 k, k_diff, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, k_diff, v))
#                 k = torch.stack([k, k_diff], dim = 1)
#             else:
#                 q = self.to_q(x)
#                 q = rearrange(q, 'b n (h d) -> b h n d', h = h)
#                 k, v = self.to_kv(kv_input).chunk(2, dim=-1)
#                 k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
#         else:
#             # Use fused linear projection
#             if self.differential:
#                 q, k, v, q_diff, k_diff = self.to_qkv(x).chunk(5, dim=-1)
#                 q, k, v, q_diff, k_diff  = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v, q_diff, k_diff))
#                 q = torch.stack([q, q_diff], dim = 1)
#                 k = torch.stack([k, k_diff], dim = 1)
#             else:
#                 q, k, v = self.to_qkv(x).chunk(3, dim=-1)
#                 q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

#         # Normalize q and k for cosine sim attention
#         if self.qk_norm == "l2":
#             q = F.normalize(q, dim=-1)
#             k = F.normalize(k, dim=-1)
#         elif self.qk_norm != "none":
#             q, k = self.apply_qk_layernorm(q, k)

#         if rotary_pos_emb is not None:
#             freqs, _ = rotary_pos_emb
#             q_dtype = q.dtype
#             k_dtype = k.dtype
#             q = q.to(torch.float32)
#             k = k.to(torch.float32)
#             freqs = freqs.to(torch.float32)
#             if q.shape[-2] >= k.shape[-2]:
#                 ratio = q.shape[-2] / k.shape[-2]
#                 q_freqs, k_freqs = freqs, ratio * freqs
#             else:
#                 ratio = k.shape[-2] / q.shape[-2]
#                 q_freqs, k_freqs = ratio * freqs, freqs
#             q = apply_rotary_pos_emb(q, q_freqs)
#             k = apply_rotary_pos_emb(k, k_freqs)
#             q = q.to(v.dtype)
#             k = k.to(v.dtype)
        
#         n, device = q.shape[-2], q.device

#         causal = self.causal if causal is None else causal

#         if n == 1 and causal:
#             causal = False

#         if self.differential:
#             q, q_diff = q.unbind(dim = 1)
#             k, k_diff = k.unbind(dim = 1)
#             out = self.apply_attn(q, k, v,  causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window)
#             out_diff = self.apply_attn(q_diff, k_diff, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window)
#             out = out - out_diff
#         else:
#             out = self.apply_attn(q, k, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window)

#         # merge heads
#         out = rearrange(out, ' b h n d -> b n (h d)')

#         # Communicate between heads
        
#         # with autocast(enabled = False):
#         #     out_dtype = out.dtype
#         #     out = out.to(torch.float32)
#         #     out = self.to_out(out).to(out_dtype)
#         out = self.to_out(out)

#         if self.feat_scale:
#             out_dc = out.mean(dim=-2, keepdim=True)
#             out_hf = out - out_dc

#             # Selectively modulate DC and high frequency components
#             out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

#         return out

# class ConformerModule(nn.Module):
#     def __init__(
#         self,
#         dim,
#         norm_kwargs = {},
#     ):     

#         super().__init__()

#         self.dim = dim
        
#         self.in_norm = LayerNorm(dim, **norm_kwargs)
#         self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
#         self.glu = GLU(dim, dim, nn.SiLU())
#         self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
#         self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
#         self.swish = nn.SiLU()
#         self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

#     #@compile
#     def forward(self, x):
#         x = self.in_norm(x)
#         x = rearrange(x, 'b n d -> b d n')
#         x = self.pointwise_conv(x)
#         x = rearrange(x, 'b d n -> b n d')
#         x = self.glu(x)
#         x = rearrange(x, 'b n d -> b d n')
#         x = self.depthwise_conv(x)
#         x = rearrange(x, 'b d n -> b n d')
#         x = self.mid_norm(x)
#         x = self.swish(x)
#         x = rearrange(x, 'b n d -> b d n')
#         x = self.pointwise_conv_2(x)
#         x = rearrange(x, 'b d n -> b n d')

#         return x

# class TransformerBlock(nn.Module):
#     def __init__(
#             self,
#             dim,
#             dim_heads = 64,
#             cross_attend = False,
#             dim_context = None,
#             global_cond_dim = None,
#             causal = False,
#             zero_init_branch_outputs = True,
#             conformer = False,
#             layer_ix = -1,
#             remove_norms = False,
#             add_rope = False,
#             layer_scale = False,
#             attn_kwargs = {},
#             ff_kwargs = {},
#             norm_kwargs = {}
#     ):
        
#         super().__init__()
#         self.dim = dim
#         self.dim_heads = min(dim_heads,dim)
#         self.cross_attend = cross_attend
#         self.dim_context = dim_context
#         self.causal = causal
       
#         if layer_scale and zero_init_branch_outputs:
#             print('zero_init_branch_outputs is redundant with layer_scale, setting zero_init_branch_outputs to False')
#             zero_init_branch_outputs = False
            
#         self.pre_norm = LayerNorm(dim,**norm_kwargs) if not remove_norms else DynamicTanh(dim)

#         self.add_rope = add_rope

#         self.self_attn = Attention(
#             dim,
#             dim_heads = self.dim_heads,
#             causal = causal,
#             zero_init_output=zero_init_branch_outputs,
#             **attn_kwargs
#         )

#         self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

#         self.cross_attend = cross_attend
#         if cross_attend:
#             self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else DynamicTanh(dim)
#             self.cross_attn = Attention(
#                 dim,
#                 dim_heads = self.dim_heads,
#                 dim_context=dim_context,
#                 causal = causal,
#                 zero_init_output=zero_init_branch_outputs,
#                 **attn_kwargs
#             )
#             self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        
#         self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else DynamicTanh(dim)
#         self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
#         self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()

#         self.layer_ix = layer_ix

#         self.conformer = None
#         if conformer:
#             self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs)
#             self.conformer_scale = LayerScale(dim) if layer_scale else nn.Identity()

#         self.global_cond_dim = global_cond_dim

#         if global_cond_dim is not None:
#             self.to_scale_shift_gate = nn.Parameter(torch.randn(6*dim)/dim**0.5)

#         self.rope = RotaryEmbedding(self.dim_heads // 2) if add_rope else None
        
#     @compile
#     def forward(
#         self,
#         x,
#         context = None,
#         global_cond=None,
#         rotary_pos_emb = None,
#         self_attention_block_mask = None,
#         self_attention_score_mod = None,
#         cross_attention_block_mask = None,
#         cross_attention_score_mod = None,
#         self_attention_flash_sliding_window = None,
#         cross_attention_flash_sliding_window = None
#     ):
#         if rotary_pos_emb is None and self.add_rope:
#             rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

#         if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
            
#             scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)

#             # self-attention with adaLN
#             residual = x
#             x = self.pre_norm(x)
#             x = x * (1 + scale_self) + shift_self
#             x = self.self_attn(x, rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window)
#             x = x * torch.sigmoid(1 - gate_self)
#             x = self.self_attn_scale(x)
#             x = x + residual

#             '''if context is not None and self.cross_attend:
#                 x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))'''
                
#             if context is not None and self.cross_attend:
#                 q_len, k_len = x.shape[-2], context.shape[-2]
#                 cross_mask = create_cross_causal_mask(q_len, k_len, x.device)

#                 x = x + self.cross_attn_scale(
#                     self.cross_attn(
#                         self.cross_attend_norm(x),
#                         context=context,
#                         causal=False,#True,                # enforce causal
#                         attn_mask=cross_mask,       # pass mask
#                         flex_attention_block_mask=cross_attention_block_mask,
#                         flex_attention_score_mod=cross_attention_score_mod,
#                         flash_attn_sliding_window=cross_attention_flash_sliding_window
#                     )
#                 )
            
#             if self.conformer is not None:
#                 x = x + self.conformer_scale(self.conformer(x))

#             # feedforward with adaLN
#             residual = x
#             x = self.ff_norm(x)
#             x = x * (1 + scale_ff) + shift_ff
#             x = self.ff(x)
#             x = x * torch.sigmoid(1 - gate_ff)
#             x = self.ff_scale(x)
#             x = x + residual

#         else:
#             x = x + self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window))

#             '''if context is not None and self.cross_attend:
#                 x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))'''
                
#             if context is not None and self.cross_attend:
#                 q_len, k_len = x.shape[-2], context.shape[-2]
#                 cross_mask = create_cross_causal_mask(q_len, k_len, x.device)

#                 x = x + self.cross_attn_scale(
#                     self.cross_attn(
#                         self.cross_attend_norm(x),
#                         context=context,
#                         causal=False,#True,                # enforce causal
#                         attn_mask=cross_mask,       # pass mask
#                         flex_attention_block_mask=cross_attention_block_mask,
#                         flex_attention_score_mod=cross_attention_score_mod,
#                         flash_attn_sliding_window=cross_attention_flash_sliding_window
#                     )
#                 )
                    
#             if self.conformer is not None:
#                 x = x + self.conformer_scale(self.conformer(x))

#             x = x + self.ff_scale(self.ff(self.ff_norm(x)))
#         return x
        
# class ContinuousTransformer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         depth,
#         *,
#         dim_in = None,
#         dim_out = None,
#         dim_heads = 64,
#         cross_attend=False,
#         cond_token_dim=None,
#         final_cross_attn_ix=-1,
#         global_cond_dim=None,
#         causal=False,
#         rotary_pos_emb=True,
#         zero_init_branch_outputs=True,
#         conformer=False,
#         use_sinusoidal_emb=False,
#         use_abs_pos_emb=False,
#         abs_pos_emb_max_length=10000,
#         num_memory_tokens=0,
#         sliding_window=None,
#         **kwargs
#         ):

#         super().__init__()

#         self.dim = dim
#         self.depth = depth
#         self.causal = causal
#         self.layers = nn.ModuleList([])

#         self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
#         self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

#         if rotary_pos_emb:
#             self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
#         else:
#             self.rotary_pos_emb = None

#         self.num_memory_tokens = num_memory_tokens
#         if num_memory_tokens > 0:
#             self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

#         self.use_sinusoidal_emb = use_sinusoidal_emb
#         if use_sinusoidal_emb:
#             self.pos_emb = ScaledSinusoidalEmbedding(dim)

#         self.use_abs_pos_emb = use_abs_pos_emb
#         if use_abs_pos_emb:
#             self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length + self.num_memory_tokens)

#         self.global_cond_embedder = None
#         if global_cond_dim is not None:
#             self.global_cond_embedder = nn.Sequential(
#                 nn.Linear(global_cond_dim, dim),
#                 nn.SiLU(),
#                 nn.Linear(dim, dim * 6)
#             )

#         self.final_cross_attn_ix = final_cross_attn_ix

#         self.sliding_window = sliding_window

#         for i in range(depth):
#             should_cross_attend = cross_attend and (self.final_cross_attn_ix == -1 or i <= (self.final_cross_attn_ix))
#             self.layers.append(
#                 TransformerBlock(
#                     dim,
#                     dim_heads = dim_heads,
#                     cross_attend = should_cross_attend,
#                     dim_context = cond_token_dim,
#                     global_cond_dim = global_cond_dim,
#                     causal = causal,
#                     zero_init_branch_outputs = zero_init_branch_outputs,
#                     conformer=conformer,
#                     layer_ix=i,
#                     **kwargs
#                 )
#             )
        
#     def forward(
#         self,
#         x,
#         prepend_embeds = None,
#         global_cond = None,
#         return_info = False,
#         use_checkpointing = True,
#         exit_layer_ix = None,
#         **kwargs
#     ):
#         batch, seq, device = *x.shape[:2], x.device

#         model_dtype = next(self.parameters()).dtype
#         x = x.to(model_dtype)

#         info = {
#             "hidden_states": [],
#         }

#         x = self.project_in(x)

#         if prepend_embeds is not None:
#             prepend_length, prepend_dim = prepend_embeds.shape[1:]

#             assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

#             x = torch.cat((prepend_embeds, x), dim = -2)

#         if self.num_memory_tokens > 0:
#             memory_tokens = self.memory_tokens.expand(batch, -1, -1)
#             x = torch.cat((memory_tokens, x), dim=1)

#         if self.rotary_pos_emb is not None:
#             rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
#         else:
#             rotary_pos_emb = None

#         if self.use_sinusoidal_emb or self.use_abs_pos_emb:
#             x = x + self.pos_emb(x)

#         if global_cond is not None and self.global_cond_embedder is not None:
#             global_cond = self.global_cond_embedder(global_cond)

#         # Iterate over the transformer layers
#         for layer_ix, layer in enumerate(self.layers):

#             if use_checkpointing:
#                 x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, self_attention_flash_sliding_window = self.sliding_window, **kwargs)
#             else:
#                 x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, self_attention_flash_sliding_window = self.sliding_window, **kwargs)

#             if return_info:
#                 info["hidden_states"].append(x)

#             if exit_layer_ix is not None and layer_ix == exit_layer_ix:
#                 x = x[:, self.num_memory_tokens:, :]

#                 if return_info:
#                     return x, info
                
#                 return x

#         x = x[:, self.num_memory_tokens:, :]

#         x = self.project_out(x)

#         if return_info:
#             return x, info
        
#         return x
