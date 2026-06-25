import numpy as np
import torch 
import typing as tp
import math 
from torchaudio import transforms as T
from torch.nn.functional import interpolate

from .utils import prepare_audio
from .sampling import sample, sample_k, sample_rf
from ..data.utils import PadCrop

def generate_diffusion_uncond(
        model,
        steps: int = 250,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor:
    print('doing unconditional!!!!!!!!!!!!!!!')
    print(f"DEBUG: sampler_kwargs: {sampler_kwargs}")
    # The length of the output in audio samples 
    audio_sample_size = sample_size
    
    _, orig_len = init_audio
    print('original length:', orig_len.shape[-1])

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    #noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    if init_audio is not None:
        print('The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.')
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            print('io channels set')
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=init_audio.shape[-1], target_channels=io_channels, device=device)
        
        print('init audio shape after prepare_audio... channels, length', init_audio.shape[1], init_audio.shape[2])

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            print('encoding init audio')
            init_audio = model.pretransform.encode(init_audio)
       
            init_audio = init_audio.repeat(batch_size, 1, 1)
            print(f"Original latent length: {init_audio.shape[-1]} samples")
            print(f"Expected preserved samples: should match original latent length")

        init_audio = init_audio.repeat(batch_size, 1, 1)
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None

    # Inpainting mask
    
    if init_audio is not None:
        # variations
        print('init audio is not none')
        sampler_kwargs["sigma_max"] = init_noise_level
        init_data = init_audio
        sample_size = init_audio.shape[-1]
        mask = inpaint_mask 
    else:
        mask = inpaint_mask

    # Now the generative AI part:
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
    diff_objective = model.diffusion_objective
    print(f"DEBUG: sampler_kwargs: {sampler_kwargs}")
    if diff_objective == "v":    
        # k-diffusion denoising process go!
        #sampled = sample_k(model.model, noise, init_audio, mask, steps, **sampler_kwargs, device=device)
        #sampler_type="dpmpp-2m"
        sampled = sample_k(model.model, noise, init_audio, steps, sampler_type)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:
        print('doing rectified flow diffusion objective')
        print('init_noise_level:', init_noise_level)
        sampled = sample_rf(model.model, noise, inpaint_mask, init_audio, steps, sampler_type="pingpong_cont", sigma_max=1, device=device, original=init_data)
    print('sampled shape:', sampled.shape)
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #print(f"DEBUG: sample returned: {sampled}")
        #print(f"DEBUG: sampled type: {type(sampled)}")
        print(f"DEBUG: sampled shape: {sampled.shape if sampled is not None else 'None'}")
        sampled = model.pretransform.decode(sampled)
        
    print(f"Final sampled shape: {sampled.shape}")
    # Save just the original part to compare
    #original_part = sampled[..., :original_latent_length]

    # Return audio
    return sampled#, original_part


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """
    
    print('________generate_diffusion_cond________')
    final_sampler_kwargs = {
        'sigma_min': 0.1,
        'sigma_max': 50,
        'rho': 1.0,
        'sampler_type': "dpmpp-2m"
    }
    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        print('latent diffusion selected')
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
    #print('DEBUG: conditioning inputs shape:', conditioning_inputs.shape)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        print('The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.')
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            print(f"Original latent length: {init_audio.shape[-1]} samples")
            print(f"Expected preserved samples: should match original latent length")

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level        

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        print('k-diffusion denoising process go!')
        sampled = sample_k(model.model, noise, init_audio, steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:
        print('doing rf')

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        #sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, dist_shift=model.dist_shift, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
        print('init_noise_level:', init_noise_level)
        sampled = sample_rf(model.model, noise, init_audio, steps, sampler_type="pingpong", sigma_max=1, **conditioning_inputs, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled

def generate_diffusion_cond_inpaint(
        model,
        steps: int = 30,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask=None,
        return_latents = False,
        sigma_min=0.1,
        sigma_max=50,
        rho=3.0,
        sampler_type="dpmpp-2m",
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion inpainting model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        inpaint_mask: A mask to use for inpainting. Shape should be [batch_size, sample_size]
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """
    print('........DEBUG: inpaint mask:', inpaint_mask.shape)
    print('........DEBUG: inpaint audio:', inpaint_audio[1].shape)
    
    final_sampler_kwargs = {
        'sigma_min': 0.1,
        'sigma_max': 50,
        'rho': 1.0,
        'sampler_type': "dpmpp-2m"
    }
    
    #print(f"DEBUG: Final sampler_kwargs: {final_sampler_kwargs}")
    
    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        print('downsampling sample size to latent size')
        sample_size = sample_size // model.pretransform.downsampling_ratio
    
    if inpaint_mask is not None:
        print('making inpaint mask a float')
        inpaint_mask = inpaint_mask.float()




    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        print('The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.')
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                print('shrinking inpaint_mask')
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=init_audio.shape[-1], mode='nearest').squeeze(1)
                print('inpaint mask shape:', inpaint_mask.shape)

        init_audio = init_audio.repeat(batch_size, 1, 1)

    if inpaint_audio is not None:
        print('preparing inpaint audio')
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        inpaint_sr, inpaint_audio = inpaint_audio
        print('inpaint audio shape:', inpaint_audio.shape)

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            #print('DEBUG 1: pretransform is not None; doing latent diffusion')
            io_channels = model.pretransform.io_channels
            #print('io channels:', io_channels)

        # Prepare the initial audio for use by the model
        print('target length:', audio_sample_size)
        inpaint_audio = prepare_audio(inpaint_audio, in_sr=inpaint_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)
        #print('DEBUG: inpaint audio prepared')
        #print('inpaint_audio shape:', inpaint_audio.shape)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            #print('DEBUG 2: pretransform is not None; doing latent diffusion') 
            inpaint_audio = model.pretransform.encode(inpaint_audio)
            
            # Interpolate inpaint mask to the same length as the encoded init audio
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=inpaint_audio.shape[-1], mode='nearest').squeeze(1)

        inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
    else:
       
        if inpaint_mask is not None:
            # interpolate inpaint mask to the sample size
            inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode='nearest').squeeze(1)

    if inpaint_mask is None:
        mask = torch.zeros((batch_size, 1, sample_size), device=device)  
    else:
        mask = inpaint_mask.unsqueeze(1)

    # Inpainting mask
    mask = mask.to(device)

    if inpaint_audio is not None:
        inpaint_input = inpaint_audio * mask.expand_as(inpaint_audio)
    else:
        inpaint_input = torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors['inpaint_mask'] = [mask]
    #print('DEBUG: mask shape', mask.shape)
    conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
    #print('DEBUG: inpaint input shape', inpaint_input.shape)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
    #print('DEBUG: conditioning inputs 1:', conditioning_inputs)

    if negative_conditioning_tensors:
        negative_conditioning_tensors['inpaint_mask'] = [mask]
        negative_conditioning_tensors['inpaint_masked_input'] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    
    if init_audio is not None:
        #print('DEBUG: init_audio is not None')
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    #print('DEBUG: conditioning inputs 2:', conditioning_inputs)
    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective
    #print(f"DEBUG: diff_objective = {diff_objective}")
    #print(f"DEBUG: sampler_type = {sampler_kwargs.get('sampler_type', 'not specified')}")
    
    if diff_objective == "v":
        print('DEBUG: diffusion objective is v')    
        # k-diffusion denoising process go!
        #print(f"DEBUG k: About to call sample_k with: sigma_min={final_sampler_kwargs.get('sigma_min')}, sigma_max={final_sampler_kwargs.get('sigma_max')}, rho={final_sampler_kwargs.get('rho')}, sampler_type={final_sampler_kwargs.get('sampler_type')}")
        sampled = sample_k(model.model, noise, init_data=init_audio, steps=steps, **final_sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:
        #print('DEBUG: diffusion objective is rf')
        print('doing rectified flow diffusion objective')
        if "sigma_min" in final_sampler_kwargs:
            del final_sampler_kwargs["sigma_min"]

        if "rho" in final_sampler_kwargs:
            del final_sampler_kwargs["rho"]
            
            

        #print(f"DEBUG rf: About to call sample_k with: sigma_min={final_sampler_kwargs.get('sigma_min')}, sigma_max={final_sampler_kwargs.get('sigma_max')}, rho={final_sampler_kwargs.get('rho')}, sampler_type={final_sampler_kwargs.get('sampler_type')}")
        #sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **final_sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
        sampled = sample_rf(model.model, noise, mask, init_audio, steps, sampler_type="pingpong_cont", sigma_max=1, device=device)
    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #print('DEBUG: model.pretransform is not None and not return_latents')
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled


# builds a softmask given the parameters
# returns array of values 0 to 1, size sample_size, where 0 means noise / fresh generation, 1 means keep the input audio, 
# and anything between is a mixture of old/new
# ideally 0.5 is half/half mixture but i haven't figured this out yet
def build_mask(sample_size, mask_args):
    maskstart = math.floor(mask_args["maskstart"]/100.0 * sample_size)
    maskend = math.ceil(mask_args["maskend"]/100.0 * sample_size)
    softnessL = round(mask_args["softnessL"]/100.0 * sample_size)
    softnessR = round(mask_args["softnessR"]/100.0 * sample_size)
    marination = mask_args["marination"]
    # use hann windows for softening the transition (i don't know if this is correct)
    hannL = torch.hann_window(softnessL*2, periodic=False)[:softnessL]
    hannR = torch.hann_window(softnessR*2, periodic=False)[softnessR:]
    # build the mask. 
    mask = torch.zeros((sample_size))
    mask[maskstart:maskend] = 1
    mask[maskstart:maskstart+softnessL] = hannL
    mask[maskend-softnessR:maskend] = hannR
    # marination finishes the inpainting early in the denoising schedule, and lets audio get changed in the final rounds
    if marination > 0:        
        mask = mask * (1-marination) 
    #print(mask)
    return mask
