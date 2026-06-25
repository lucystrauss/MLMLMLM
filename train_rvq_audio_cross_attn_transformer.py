import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from mappers.mapper_models import ar_loss, flatten_and_offset_latents, unflatten_and_unoffset_tokens, ARMapper, sample_ar_w_temp_cond_causal
from einops import rearrange
from stable_audio_tools.models import create_model_from_config
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.paired_dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config

from stable_audio_tools.models.transformer import ContinuousTransformer

from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training.losses.auraloss import MultiResolutionSTFTLoss
from stable_audio_tools.models.discriminators import EncodecDiscriminator


# wandb logging:
import wandb
import random

import matplotlib.pyplot as plt

# Disable FlashAttention and memory‑efficient kernels
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Force math implementation (always available, slower but safe)
torch.backends.cuda.enable_math_sdp(True)

mapper_lr=1e-3
num_mapper_epochs=100


wandb.login()
wandb_project = "my-awesome-project"

wandb_config = {
            'num_epochs': 50,
            'lr': 1e-3,
            'hidden_dim': 256,
            'batch_size': 4,
        }
    
        
def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b

def train_mapper(mapper, train_loader, emg_enc, audio_enc, device='cuda', num_epochs=num_mapper_epochs, lr=mapper_lr, plot_every=1, wandb_config=wandb_config):
    if wandb_config is None:
        wandb_config = {
            'num_epochs': 100,
            'lr': 1e-3,
            'hidden_dim': 256,
            'batch_size': 4,
        }
    num_epochs = wandb_config['num_epochs']
    lr = wandb_config['lr']
    mapper = mapper.to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs, pct_start=0.15)
    
    
    
    #stft_loss = MultiResolutionSTFTLoss()
    
    
    emg_enc = emg_enc.to(device)
    audio_enc = audio_enc.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
    
        teacher_forcing_ratio = max(0.1, 1.0 - (epoch / num_epochs))
    
        plt_batch_idx = random.randint(0, len(train_loader) - 1)
        
        print(f"\n=== EPOCH {epoch} ===")
        print(f"Will plot batch index: {plt_batch_idx}")
        
        # Training
        #mapper.train()
        #emg_enc.train()
        #audio_enc.train()
        
        #print(audio_enc.state_dict().keys())
        #print(emg_enc.state_dict().keys())
        
        for batch_idx, batch in enumerate(train_loader):
        
            #audio_enc.train()
            print('batch_idx', batch_idx, flush=True)
            
            reals, info_list = batch
            reals = reals.to(device)
        
        
            
            # --- AUDIO PATH ---
            _, audio_info = audio_enc.encode(reals[:, 0:1], return_info=True)
            audio_latents = audio_info["quantizer_indices"]
            #B, T, L = audio_latents.shape
            audio_tokens = flatten_and_offset_latents(audio_latents, codebook_size=2048)
            prefix_len = 10
            start_tokens = audio_tokens[:, :prefix_len]
            
            inputs = audio_tokens[:, :-1]   # teacher forcing inputs
            targets = audio_tokens[:, 1:]

            # --- SENSOR PATH ---
            _, sensor_info = emg_enc.encode(reals[:, 1:7], return_info=True)
            sensor_latents = sensor_info["quantizer_indices"]
            sensor_tokens = flatten_and_offset_latents(sensor_latents, codebook_size=2048)
            sensor_emb = mapper.embed(sensor_tokens)  # conditioning embeddings

            
            logits = mapper(inputs, cond=sensor_emb)
            transformer_loss = ar_loss(logits, targets)
            
           

            # Backward pass
            optimizer.zero_grad()
            transformer_loss.backward()
            #torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0) # gradient clipping
            optimizer.step()
            scheduler.step()
                
            
            # Log every batch
            wandb.log({
                'epoch': epoch + 1,
                #'batch': batch_idx,
                'transformer_loss': transformer_loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Plot for the randomly selected batch
            if epoch % plot_every == 0 and batch_idx == 0:#plt_batch_idx:
                mapper.eval()
            
                with torch.no_grad():
                    # --- AUDIO PATH ---
                    _, info = audio_enc.encode(reals[:, 0:1], return_info=True)
                    latents = info["quantizer_indices"]
                    B, T, L = latents.shape
                    audio_tokens = flatten_and_offset_latents(latents, codebook_size=2048)
    
                    # --- SENSOR PATH ---
                    _, sensor_info = emg_enc.encode(reals[:, 1:7], return_info=True)
                    sensor_latents = sensor_info["quantizer_indices"]
                    sensor_tokens_full = flatten_and_offset_latents(sensor_latents, codebook_size=2048)
    
                    # Start tokens
                    prefix_len = 25
                    start_audio = audio_tokens[:, :prefix_len]
                    start_sensor = sensor_tokens_full[:, :prefix_len]
    
                    # Sensor token generator (ground truth for validation)
                    def get_next_sensor_token(step):
                        return sensor_tokens_full[:, step:step+1]
    
                    # Generate with truly causal conditioning
                    generated_audio, generated_sensor = sample_ar_w_temp_cond_causal(
                        mapper,
                        start_audio_tokens=start_audio,
                        start_sensor_tokens=start_sensor,
                        sensor_token_generator=get_next_sensor_token,
                        seq_len=audio_tokens.size(1),
                        temperature=0.8,
                        top_k=50,
                        top_p=0.95,
                        vocab_size=8192
                    )
                    # Convert back to latents
                    latents = unflatten_and_unoffset_tokens(
                        generated_audio,
                        T=audio_latents.shape[1],
                        num_quantizers=audio_latents.shape[2],
                        codebook_size=2048
                    )
                    audio_recon = audio_enc.decode_tokens(latents)
                    print(audio_recon.shape, audio_recon.device, audio_recon.dtype)
                    #print('audio_latents.shape before decode tokens', audio_latents.shape, flush=True)
                    #audio_latents = audio_latents.permute(0, 2, 1)
                    #print('audio_latents.shape after decode tokens', audio_latents.shape, flush=True)
                    #audio_recon = audio_enc.decoder(audio_latents)
                    
                    
                    
                    
                    
                print('AUDIO RECON SHAPE:', audio_recon.shape, flush=True) # (batch, channels, samples)
                reals = reals[0:1, 0:1]
                fakes = audio_recon[0:1, 0:1]
                fakes, reals = trim_to_shortest(fakes, reals)
                print('fakes.shape', fakes.shape, flush=True)
                print('reals.shape', reals.shape, flush=True)
                fakes = fakes.detach().cpu().squeeze(0)
                reals = reals.detach().cpu().squeeze(0)
                print(reals.device)
                print(fakes.device)
                print('fakes.shape', fakes.shape, flush=True)
                print('reals.shape', reals.shape, flush=True)
                reals = reals / reals.abs().max()
                fakes = fakes / fakes.abs().max()
                reals_fakes = torch.cat((reals, fakes), dim=1)
                print('REALS FAKES SHAPE:', reals_fakes.shape, flush=True)
                
                   
                torch.save(mapper.state_dict(), f'rvq_audio_transformer_causal_cross_8dec2025_{epoch+1}.pt')
                
            
                
                filename = f'recon_{epoch}_{batch_idx}.wav'
                reals_fakes = reals_fakes / reals_fakes.abs().max()
                reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, reals_fakes, 44100)
                wandb.log({"cross_modal_audio_recon": wandb.Audio(filename, sample_rate=44100)})

                mapper.train()
                audio_enc.eval()
                
                
                       
        print(f"Epoch {epoch+1}/{num_epochs} completed")
    
    return mapper

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb.login()

    args = get_all_args()
    seed = args.seed

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_loader = create_dataloader_from_config(
        dataset_config,
        batch_size=8,#args.batch_size,
        num_workers=args.num_workers,
        sample_rate=44100,
        sample_size=242550,
        audio_channels=1,
    )

    #args_dict.update({"dataset_config": dataset_config})

    emg_enc_pth = './emg_unwrapped_monday.ckpt'
    audio_enc_pth = './audio_unwrapped_monday.ckpt'
    #emg_enc_pth = './emg_rvq_unwrapped.ckpt'
    #audio_enc_pth = './audio_rvq_unwrapped.ckpt'

    #print('trying to load emg config')
    with open('./stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae_emg_rvq.json') as f:
        enc_emg_cfg = json.load(f)
        emg_enc = create_model_from_config(enc_emg_cfg)

    with open('./stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae_audio_rvq.json') as f:
        enc_audio_cfg = json.load(f)
        audio_enc = create_model_from_config(enc_audio_cfg)

    #audio_enc = loadVAEenc(audio_enc_pth, enc_audio)
    #emg_enc = loadVAEenc(emg_enc_pth, enc_emg)
    
    copy_state_dict(audio_enc, load_ckpt_state_dict(audio_enc_pth))
    copy_state_dict(emg_enc, load_ckpt_state_dict(emg_enc_pth))
    
    print("EMG encoder first layer weight checksum:", emg_enc.state_dict()['encoder.layers.0.weight_g'].sum().item(), flush=True)
    print("Audio encoder first layer weight checksum:", audio_enc.state_dict()['encoder.layers.0.weight_g'].sum().item(), flush=True)
    
    audio_enc.eval()
    emg_enc.eval()
    

    for param in emg_enc.parameters():
        param.requires_grad = False

    for param in audio_enc.parameters():
        param.requires_grad = False

    print('loaded audio & emg encoders', flush=True)


#--------------------------------------


    # Example dimensions
    dim_source = 64
    dim_target = 64
    num_samples = 10000
    batch_size = 16

    wandb_config = {
        'dim_source': dim_source,
        'dim_target': dim_target,
        'num_epochs': 100,
        'lr': 1e-3,
        'batch_size': batch_size,
        'num_samples': num_samples,
    }
    

    project = "rvq_cross_attn_transformer"
    with wandb.init(project=project, config=wandb_config) as run:


        
        #mapper = ARMapper(vocab_size=8192, d_model=512, n_layers=8, n_heads=8).to(device)
        mapper = ARMapper(vocab_size=8192, d_model=384, n_layers=6, n_heads=6).to(device)
        
        for param in mapper.parameters():
            param.requires_grad = True
        mapper.train()
        
        # Train
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
 
        #emg_enc=None
        trained_mapper = train_mapper(
            mapper, 
            train_loader, 
            emg_enc,
            audio_enc,
            device=device,
            num_epochs=num_mapper_epochs,
            lr=mapper_lr,
            plot_every=1,
            wandb_config=wandb_config
        )
        


if __name__ == "__main__":
    main()
