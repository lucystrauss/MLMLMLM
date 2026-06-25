import torch
import torch.nn as nn
import math
import time
import torch.nn.functional as F
from stable_audio_tools.models.transformer import ContinuousTransformer


def ar_loss(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    
    
def flatten_and_offset_latents(latents, codebook_size=2048):
    """
    latents: [B, T, L] tensor of indices from RVQ-VAE
    codebook_size: size of each quantizer's codebook (e.g. 2048)

    Returns:
        tokens: [B, T*L] flattened and offset indices
    """
    B, T, L = latents.shape
    offsets = torch.arange(L, device=latents.device) * codebook_size
    # Add offsets per quantizer level
    latents_offset = latents + offsets.view(1, 1, L)
    # Flatten levels into one long sequence
    tokens = latents_offset.view(B, T * L)
    return tokens
    
def unflatten_and_unoffset_tokens(tokens, T, num_quantizers=4, codebook_size=2048):
    """
    tokens: [B, T*L] flattened + offset indices
    T: number of timesteps
    Returns: [B, T, L] indices per quantizer
    """
    B, TL = tokens.shape
    L = num_quantizers
    assert TL == T * L

    # Reshape back
    tokens = tokens.view(B, T, L)

    # Remove offsets
    offsets = torch.arange(L, device=tokens.device) * codebook_size
    indices = tokens - offsets.view(1, 1, L)

    return indices
    
    
 
class ARMapper(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8, cond_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        self.transformer = ContinuousTransformer(
            dim=d_model,
            depth=n_layers,
            dim_heads=64,
            causal=True,
            cross_attend=True,
            cond_token_dim=d_model
        )

        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tokens,
        cond,
        past_kvs=None,
        past_kvs_cross=None,
        use_cache=False,
        start_pos=0,
        debug_kv=False
    ):
        """
        Now fully supports BOTH self-attention and cross-attention KV caching.
        """

        x = self.embed(tokens)

        if not use_cache:
            # Normal full-sequence inference
            x = self.transformer(x, context=cond)
            logits = self.head(x)
            return logits

        # --- Cache mode ---
        x, new_kvs, new_kvs_cross = self.transformer(
            x,
            context=cond,
            past_kvs=past_kvs,
            past_kvs_cross=past_kvs_cross,
            use_cache=True,
            start_pos=start_pos,
            debug_kv=debug_kv
        )

        logits = self.head(x)
        return logits, new_kvs, new_kvs_cross
        
def sample_ar_w_temp_cond_causal(
    model,
    seq_len,
    vocab_size,
    start_audio_tokens,
    start_sensor_tokens,  # <-- Changed: start with prefix
    sensor_token_generator,  # <-- Function to get next sensor token
    device="cuda",
    temperature=1.0,
    top_k=None,
    top_p=None
):
    """
    Autoregressively sample with truly causal conditioning.
    Both audio and sensor grow token-by-token in lockstep.
    """
    model.eval()
    audio_tokens = start_audio_tokens.to(device)
    sensor_tokens = start_sensor_tokens.to(device)

    with torch.no_grad():
        for step in range(seq_len - audio_tokens.shape[1]):
            # Embed only the sensor tokens we have so far
            sensor_emb = model.embed(sensor_tokens)  # [B, current_len, d]
            
            # Generate next audio token
            logits = model(audio_tokens, cond=sensor_emb)
            next_logits = logits[:, -1, :]

            # Apply temperature
            scaled_logits = next_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            # Top-k filtering
            if top_k is not None:
                values, indices = torch.topk(probs, top_k, dim=-1)
                mask = torch.zeros_like(probs)
                mask.scatter_(1, indices, values)
                probs = mask / mask.sum(dim=-1, keepdim=True)

            # Top-p filtering
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs <= top_p
                mask[..., 0] = True
                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(1, sorted_indices, sorted_probs * mask)
                probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

            # Sample next audio token
            next_audio_token = torch.multinomial(probs, num_samples=1)
            audio_tokens = torch.cat([audio_tokens, next_audio_token], dim=1)
            
            # Get next sensor token (from ground truth or another generator)
            next_sensor_token = sensor_token_generator(step + start_audio_tokens.shape[1])
            sensor_tokens = torch.cat([sensor_tokens, next_sensor_token], dim=1)

    return audio_tokens, sensor_tokens



def sample_ar_w_temp_cond_kv(
    model,
    tokens,
    cond_full,
    past_kvs=None,
    past_kvs_cross=None,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=1.0,
    device="cuda",
    debug_kv=False
):
    """Generate ONE next token, with correct caching for both self & cross attention."""

    model.eval()


    step_in = tokens[:, -1:].to(device)


    cond_step = cond_full[:, :tokens.size(1), :]


    start_pos = 0
    if past_kvs is not None and past_kvs[0] is not None:
        start_pos = past_kvs[0][0].shape[-2]

    if debug_kv:
        print(f"[SAMPLER] start_pos={start_pos}")
        if past_kvs is not None and past_kvs[0] is not None:
            print("[SAMPLER] past self-KV:", past_kvs[0][0].shape)
        if past_kvs_cross is not None and past_kvs_cross[0] is not None:
            print("[SAMPLER] past cross-KV:", past_kvs_cross[0][0].shape)

    with torch.no_grad():


        logits, new_kvs, new_kvs_cross = model(
            step_in,
            cond_step,
            past_kvs=past_kvs,
            past_kvs_cross=past_kvs_cross,
            use_cache=True,
            start_pos=start_pos,
            debug_kv=debug_kv
        )

        next_logits = logits[:, -1, :]


        if repetition_penalty and repetition_penalty > 1.0:
            used = set(tokens[0].tolist())
            if used:
                next_logits[:, list(used)] /= repetition_penalty


        next_logits /= max(temperature, 1e-6)
        probs = F.softmax(next_logits, dim=-1)


        if top_k is not None:
            v, idx = torch.topk(probs, top_k)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, idx, v)
            probs = mask / mask.sum(-1, keepdim=True)


        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_idx = probs.sort(descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True

            filtered = torch.zeros_like(probs)
            filtered.scatter_(1, sorted_idx, sorted_probs * keep)
            probs = filtered / filtered.sum(-1, keepdim=True)

        next_token = torch.multinomial(probs, 1)



    return next_token, (new_kvs, new_kvs_cross)



def unflatten_and_unoffset_token_by_token(tokens, T=None, num_quantizers=4, codebook_size=2048):
    """
    tokens: [B, T*L] flattened + offset indices
    T: number of timesteps (optional). If None, inferred from tokens length.
    Returns: [B, T, L] indices per quantizer
    """
    B, TL = tokens.shape
    L = num_quantizers

    # Infer T if not provided
    if T is None:
        assert TL % L == 0, f"Token length {TL} not divisible by num_quantizers {L}"
        T = TL // L
    else:
        # Safety check if T is provided
        assert TL == T * L, f"Provided T={T} does not match tokens length {TL} with L={L}"

    # Reshape back
    tokens = tokens.view(B, T, L)

    # Remove offsets
    offsets = torch.arange(L, device=tokens.device) * codebook_size
    indices = tokens - offsets.view(1, 1, L)

    return indices
    
