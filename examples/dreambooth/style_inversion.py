from __future__ import annotations
from typing import Callable
from JDiffusion import StableDiffusionPipeline
# from JDiffusion.pipelines.pipeline_stable_diffusion_jittor import rescale_noise_cfg
import torch
import jittor as jt
from tqdm import tqdm
import numpy as np
from utils import to_tensor


T = jt.Var
TN = T
InversionCallback = Callable[[StableDiffusionPipeline, int, T, dict[str, T]], dict[str, T]]


def _encode_text_sd(prompt: str,  model: StableDiffusionPipeline, negative_prompt=None,):
    text_inputs = model.tokenizer(
        prompt,
        padding='max_length',
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    )
    text_input_ids = text_inputs.input_ids

    if hasattr(model.text_encoder.config, "use_attention_mask") and model.text_encoder.config.use_attention_mask:
        attention_mask = uncond_input.attention_mask
    else:
        attention_mask = None

    with jt.no_grad():
        prompt_embeds = model.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

    if negative_prompt is None:
        uncond_tokens = [""]
    elif isinstance(negative_prompt, str):
        uncond_tokens = [negative_prompt]

    max_length = prompt_embeds.shape[1]
    uncond_input = model.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    negative_prompt_embeds = model.text_encoder(
        uncond_input.input_ids,
        attention_mask=attention_mask,
    )
    negative_prompt_embeds = negative_prompt_embeds[0]
    return jt.concat([negative_prompt_embeds, prompt_embeds])


def _encode_image(image: np.ndarray, model: StableDiffusionPipeline) -> T:
    image = to_tensor(image)
    if image.size()[0] == 1:
        image = jt.concat([image]*3, dim=0)
    latent = model.vae.encode(image.unsqueeze(0).to(model.vae.device) * 2 - 1)['latent_dist'].mean * model.vae.config.scaling_factor
    return latent


def _next_step(model: StableDiffusionPipeline, model_output: T, timestep: int, sample: T) -> T:
    timestep, next_timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = model.scheduler.alphas_cumprod[int(timestep)] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[int(next_timestep)]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def _get_noise_pred(
    model: StableDiffusionPipeline,
    latent: T,
    t: T,
    text_embeds: T,
    guidance_scale: float,
    ):
    latents_input = jt.concat([latent]*2)
    latents_input = model.scheduler.scale_model_input(latents_input, t)
    timestep_cond = None
    if model.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = jt.array(guidance_scale - 1).repeat(1)
        timestep_cond = model.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=model.unet.config.time_cond_proj_dim
        ).to(dtype=latent.dtype)
    noise_pred = model.unet(
                    latents_input,
                    t,
                    encoder_hidden_states=text_embeds,
                    timestep_cond=timestep_cond,
                    return_dict=False,
                )[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def _ddim_loop(model: StableDiffusionPipeline, z0, prompt, guidance_scale) -> T:
    all_latent = [z0]
    text_embeds = _encode_text_sd(
        prompt=prompt,
        model=model,
    )
    latent = z0
    for i in tqdm(range(model.scheduler.num_inference_steps)):
        t = int(model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1])
        noise_pred = _get_noise_pred(model, latent, t, text_embeds,
                                     guidance_scale)
        latent = _next_step(model=model, model_output=noise_pred, timestep=t, sample=latent)
        all_latent.append(latent)
    return jt.concat(all_latent).flip(0)


def make_inversion_callback(zts, offset: int = 0) -> [T, InversionCallback]:

    def callback_on_step_end(pipeline, i: int, t: T, callback_kwargs: dict[str, T]) -> dict[str, T]:
        latents = callback_kwargs['latents']
        latents[0] = zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        return {'latents': latents}
    return  zts[offset], callback_on_step_end



def make_content_style_inversion_callback(content_zts, style_zts, offset=0, cs_injection=None) -> InversionCallback:
    def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        # assert (len(size) == 4)
        C = size[:1][0]
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)
        return feat_mean, feat_std
    def adain(cnt_feat, sty_feat):
        cnt_mean, cnt_std = calc_mean_std(cnt_feat)
        sty_mean, sty_std = calc_mean_std(sty_feat)
        output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
        return output
    def callback_on_step_end(pipeline, i: int, t: T, callback_kwargs: dict[str, T]) -> dict[str, T]:
        latents = callback_kwargs['latents']
        latents[0] = style_zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        latents[1] = content_zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        if cs_injection is not None:
            if i == 1:
                latents[2] = adain(content_zts[cs_injection].to(latents.device, latents.dtype), style_zts[cs_injection].to(latents.device, latents.dtype))
        return {'latents': latents}
    return callback_on_step_end



@torch.no_grad()
def ddim_inversion(model: StableDiffusionPipeline, x0: np.ndarray, prompt: str, num_inference_steps: int, guidance_scale,) -> T:
    z0 = _encode_image(x0, model, )
    model.scheduler.set_timesteps(num_inference_steps, device=z0.device)
    zs = _ddim_loop(model, z0, prompt, guidance_scale)
    return zs