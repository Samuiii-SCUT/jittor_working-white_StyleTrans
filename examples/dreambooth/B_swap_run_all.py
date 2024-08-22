import json, os, tqdm
os.environ['disable_lock']='1'
import jittor as jt
import style_inversion as inver
from JDiffusion.pipelines import StableDiffusionPipeline
from JDiffusion.utils import randn_tensor
from diffusers import DPMSolverMultistepScheduler
import numpy as np
from utils import *
import time

max_num = 28
dataset_root = "B"
use_share_attn = False
use_swap_query = True
activate_layer_indices = [[24, 30]]
activate_step_indices = [[10, 49]]
inversion_guidance = 3.5
generate_guidance = 7.5
out_root = f"./result"
k = 5
weight_prefix = "./checkpoints/styleB_v1"

def inversion(taskid, pipe, num_inference_steps=50, guidance_scale=3.5, ref_name=None):
    task_dir = os.path.join(dataset_root, taskid, 'images')
    img_file = ref_name+'.png'
    img_path = os.path.join(task_dir, img_file)
    x0 = load_image(img_path)
    with open(os.path.join(dataset_root, taskid, 'train_prompt.json'), 'r') as f:
        src_prompt = json.load(f)[img_file[:-4]]
    zts = inver.ddim_inversion(model=pipe, x0=x0, prompt=src_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    zt, inversion_callback = inver.make_inversion_callback(zts, offset=0)
    return src_prompt, zt, inversion_callback, img_file[:-4]

with open(f"lhw_my.json", "r") as file:
    final_res = json.load(file)
with open(f"lhw_my_seeds.json", "r") as file:
    f_seeds = json.load(file)

def swap_run():
    with jt.no_grad():
        for taskid in tqdm.tqdm(final_res.keys()):
            loras = list(final_res[taskid].keys())

            for lora_file in loras:
                print(lora_file)
                pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.scheduler.fix_traj_t_start = 150
                pipe.scheduler.fix_traj_t_end = 150 - 50
                pipe.load_lora_weights(f"{weight_prefix}_{taskid}/{lora_file}.bin")

                str_activate_layer, str_activate_step = pipe.activate_layer(
                        activate_layer_indices=activate_layer_indices, 
                        attn_map_save_steps=[], 
                        activate_step_indices=activate_step_indices,
                        use_shared_attention=use_share_attn,
                        adain_queries=use_swap_query,
                        style_img_index=[0]
                    )

                for ref in final_res[taskid][lora_file].keys():
                    prompts_seeds = f_seeds[taskid][lora_file][ref]
                    # prompts_seeds = final_res[taskid][lora_file][ref]
                    final_prompts = list(final_res[taskid][lora_file][ref].keys())
                    print(final_prompts)
                    src_prompt, zt, inversion_callback, src_subject = inversion(taskid, pipe, num_inference_steps=50, guidance_scale=inversion_guidance, ref_name=ref)
                    
                    out_dir = os.path.join(out_root, taskid)

                    # 如果使用多个seed，这步必要
                    zt = jt.unsqueeze(zt, dim=0)

                    subjects = list(prompts_seeds.keys())
                    print(subjects)
                    seeds = list(prompts_seeds.values())
                    print(seeds)
                    ps = [id2style_old[taskid].replace("{obj}", p.lower()) for p in subjects]
                    # 带 aligned transfer 的生成
                    length = 0
                    for i in range(0, len(ps), k):
                        if length == len(final_prompts):
                            break
                        set_p = ps[i:i+k]
                        if final_prompts[length] not in subjects[i:i+k]:
                            continue
                        shape = list(zt.shape)[1:]
                        # shape.insert(0, 1+k)
                        # seed = [i for i in range(1, len(set_p)+1)]
                        latents = randn_tensor(shape, seed=seeds[i:i+k], dtype=pipe.unet.dtype).to('cuda')
                        latents = jt.concat([zt, latents], dim=0)
                        set_p.insert(0, src_prompt)
                        print(set_p)
                        images = pipe(set_p, latents=latents, num_inference_steps=50, width=512, height=512, guidance_scale=generate_guidance, callback_on_step_end=inversion_callback).images
                        os.makedirs(out_dir, exist_ok=True)
                        for j in range(1, len(images)):
                            if subjects[i+j-1] not in final_prompts:
                                continue
                            img_path = os.path.join(out_dir, f"{subjects[i+j-1]}.png")
                            images[j].save(img_path)
                            print(f"{subjects[i+j-1]}.png saved.")
                            length += 1
