import json, os, tqdm
os.environ['disable_lock']='1'
import jittor as jt
import style_inversion as inver
from JDiffusion.pipelines import StableDiffusionPipeline
from JDiffusion.utils import randn_tensor
from diffusers import DDIMScheduler
import numpy as np
from utils import *
import time
from B_get_all_content_ref import get_content

max_num = 28
dataset_root = "B"
use_share_attn = False
use_swap_query = True
swap_str = 'my_swap' if use_swap_query else 'swap'
activate_layer_indices = [[24, 30]]
content_activate_layer_indices = [[0, 12]]
activate_step_indices = [[10, 49]]
content_activate_step_indices = [[0, 30]]
inversion_guidance = 4
generate_guidance = 7.5
out_root = f"./result"
k = 1
use_origin_sd = False
weight_prefix = "./checkpoints/styleB_v1"
weight_prefix_v2 = "./checkpoints/styleB_v2"
content_ref_dir = "B_ref"


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


def content_inversion(taskid, pipe, num_inference_steps=50, guidance_scale=3.5, save_root=None, prompts=None):
    get_content()
    # Content inversion
    content_dir = os.path.join(content_ref_dir, taskid)
    content_img_files = os.listdir(content_dir)
    content_img_files.sort()
    content_zts_list = []
    for c_img_file in content_img_files:
        if prompts is not None:
            if c_img_file[:-4] not in prompts:
                continue
        img_path = os.path.join(content_dir, c_img_file)
        c_x0 = load_image(img_path, is_grey=True)
        c_src_prompt = id2style_v2[taskid].replace('{obj}', c_img_file[:-4])
        content_zts = inver.ddim_inversion(model=pipe, x0=c_x0, prompt=c_src_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        content_zts_list.append(content_zts)
    return content_zts_list


def style_inversion(taskid, pipe, num_inference_steps=50, guidance_scale=3.5, ref_name=None, save_root=None):
    # Style inversion
    task_dir = os.path.join(dataset_root, taskid, 'images')
    img_file = ref_name+'.png'
    img_path = os.path.join(task_dir, img_file)
    x0 = load_image(img_path)
    with open(os.path.join(dataset_root, taskid, 'train_prompt.json'), 'r') as f:
        src_prompt = json.load(f)[img_file[:-4]]
    zts = inver.ddim_inversion(model=pipe, x0=x0, prompt=src_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    return src_prompt, zts, img_file[:-4]

kn = ['19', '26', '04', '06', '12', '13', '05', '25']

with open(f"lhw_f.json", "r") as file:
    final_res = json.load(file)

def content_swap():
    with jt.no_grad():
        for taskid in tqdm.tqdm(final_res.keys()):
            loras = list(final_res[taskid].keys())

            for lora_file in loras:
                c_prompts = []
                for ref in final_res[taskid][lora_file].keys():
                    for p in final_res[taskid][lora_file][ref].keys():
                        c_prompts.append(p)
                pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                pipe.scheduler.fix_traj_t_start = 150
                pipe.scheduler.fix_traj_t_end = 150 - 50
                if taskid in kn and int(lora_file)<2000:
                    pipe.load_lora_weights(f"{weight_prefix_v2}_{taskid}/{lora_file}.bin")
                else:
                    pipe.load_lora_weights(f"{weight_prefix}_{taskid}/{lora_file}.bin")

                print("Content Inversion")
                content_zts_list = content_inversion(taskid, pipe, num_inference_steps=50, guidance_scale=inversion_guidance, prompts=c_prompts)
                print("Style Inversion")
                style_list = []
                for ref_name in final_res[taskid][lora_file].keys():
                    style_prompt, style_zts, style_subject = style_inversion(taskid, pipe, num_inference_steps=50, guidance_scale=inversion_guidance, ref_name=ref_name)
                    style_list.append((style_prompt, style_zts, style_subject))

                pipe.activate_layer_v2(
                        activate_layer_indices,
                        content_activate_layer_indices=content_activate_layer_indices,
                        attn_map_save_steps=[],
                        total_steps = 50,
                        activate_step_indices = activate_step_indices,
                        content_activate_step_indices = content_activate_step_indices,
                        adain_queries=False,
                        tau=1.5,
                        gamma=0.75,
                        style_img_index=[0],
                        content_img_index=[1]
                        )

                
                for ref_index in range(len(style_list)):
                    style_prompt, style_zts, style_subject = style_list[ref_index]
                    out_dir = os.path.join(out_root, taskid)

                    # prompt初始化
                    subjects = list(final_res[taskid][lora_file][style_subject].keys())
                    subjects.sort()
                    ps = [id2style_v2[taskid].replace("{obj}", p.lower()) for p in subjects]
                    ps.sort()

                    # latent初始化
                    latents = randn_tensor([4, 64, 64], seed=[0, 0]+list(range(100, 100*k+1, 100)), dtype=pipe.unet.dtype).to('cuda')
                    latents[0] = style_zts[0]
                    
                    for i in range(0, len(ps)):
                        print(f"-----------------sampling: {ps[i]}-----------------")
                        content_zts = content_zts_list[i]
                        set_p = [style_prompt] + [ps[i] for j in range(k+1)]
                        inversion_callback = inver.make_content_style_inversion_callback(content_zts, style_zts, offset=0, cs_injection=None)

                        latents[1] = content_zts[0]
                        images = pipe(set_p, latents=latents, num_inference_steps=50, width=512, height=512, guidance_scale=generate_guidance, callback_on_step_end=inversion_callback).images
                        os.makedirs(out_dir, exist_ok=True)
                        for j in range(2, len(images)):
                            img_path = os.path.join(out_dir, f"{subjects[i]}.png")
                            images[j].save(img_path)

