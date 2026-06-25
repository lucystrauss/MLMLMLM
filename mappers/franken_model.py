import torch
import json

# import stable-audio-tools dataloaders, model methods, etc:
from stable_audio_tools.models import create_model_from_config
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.emg_dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model




def loadFrankenVAE(enc_ckpt, dec_ckpt, model):
    # Load both saved models
    enc_ckpt = torch.load(enc_ckpt)
    dec_ckpt = torch.load(dec_ckpt)


    # Get the state dicts
    state_dict1 = enc_ckpt["state_dict"]
    state_dict2 = dec_ckpt["state_dict"]

    # Filter encoder weights from model1 and decoder weights from model2
    encoder_state = {k: v for k, v in state_dict1.items() if k.startswith("encoder.")}
    decoder_state = {k: v for k, v in state_dict2.items() if k.startswith("decoder.")}

    # Combine them
    combined_state = {**encoder_state, **decoder_state}


    missing_keys, unexpected_keys = model.load_state_dict(combined_state, strict=False)
    print("Loaded Keys:", model.state_dict().keys())
    print("Missing:", missing_keys)
    print("Unexpected:", unexpected_keys)

    return model


def loadVAEenc(ckpt, model):

    ckpt = torch.load(ckpt)
    
    state_dict = ckpt["state_dict"]

    # Filter encoder weights from model1 and decoder weights from model2
    encoder_state = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}


    missing_keys, unexpected_keys = model.load_state_dict(encoder_state, strict=False)
    print("Loaded Keys:", model.state_dict().keys())
    print("Missing:", missing_keys)
    print("Unexpected:", unexpected_keys)

    return model