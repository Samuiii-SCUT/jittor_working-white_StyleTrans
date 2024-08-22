import json, os, tqdm
os.environ['disable_lock']='1'
import jittor as jt
from JDiffusion.pipelines import StableDiffusionPipeline
from JDiffusion.utils import randn_tensor
from diffusers import DDIMScheduler
import numpy as np
from utils import *
import style_inversion as inver
import time

max_num = 28
dataset_root = "B"
out_root = f"./B_ref"


def inversion(taskid, pipe, num_inference_steps=50, guidance_scale=3.5, img_index=0):
    task_dir = os.path.join(dataset_root, taskid, 'images')
    img_file = os.listdir(task_dir)[img_index]
    img_path = os.path.join(task_dir, img_file)
    x0 = load_image(img_path)
    with open(os.path.join(dataset_root, taskid, 'train_prompt.json'), 'r') as f:
        src_prompt = json.load(f)[img_file[:-4]]
    zts = inver.ddim_inversion(model=pipe, x0=x0, prompt=src_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    zt, inversion_callback = inver.make_inversion_callback(zts, offset=0)
    return src_prompt, zt, inversion_callback, img_file[:-4]


with open('content_imgs.json', 'r') as f:
    contents = json.load(f)

def get_content():
    with jt.no_grad():
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        pipe.load_lora_weights(f"./checkpoints/styleB_v1_18/1000.bin")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.fix_traj_t_start = 150
        pipe.scheduler.fix_traj_t_end = 150 - 50
        str_activate_layer, str_activate_step = pipe.activate_layer(
            activate_layer_indices=[[24, 30]], 
            attn_map_save_steps=[], 
            activate_step_indices=[[5, 49]],
            use_shared_attention=False,
            adain_queries=False,
        )
        for taskid in contents.keys():
        
            out_dir = os.path.join(out_root, f"{taskid}/")
            os.makedirs(out_dir, exist_ok=True)
            exist_ps = [p[:-4] for p in os.listdir(out_dir)]
            src_prompt, zt, inversion_callback, src_subject = inversion("18", pipe, num_inference_steps=50, guidance_scale=4, img_index=0)
            zt = jt.unsqueeze(zt, dim=0)

            subjects = contents[taskid]
            ps = [id2style['18'].replace("{obj}", s.lower()) for s in subjects]
            k = 1
            for i in range(0, len(ps), k):
                set_p = ps[i:i+k]
                if set_p[0] in exist_ps:
                    continue
                shape = list(zt.shape)[1:]
                seed = [i  for i in range(1, len(set_p)+1)]
                latents = randn_tensor(shape, seed=seed, dtype=pipe.unet.dtype).to('cuda')
                latents = jt.concat([zt, latents], dim=0)
                set_p.insert(0, src_prompt)
                
                # print(f"-----------------------------------{seed}-------------------------------------------")
                print(set_p)
                images = pipe(set_p, latents=latents, num_inference_steps=50, width=512, height=512, guidance_scale=7.5, callback_on_step_end=inversion_callback).images
                for j in range(1, len(images)):
                    img_path = os.path.join(out_dir, f"{subjects[i+j-1]}.png")
                    images[j].save(img_path)
