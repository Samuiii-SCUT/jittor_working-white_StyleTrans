import json, os, tqdm
import jittor as jt
from utils import init_latent_jt

from PIL import Image

def create_number_list(n):
    return list(range(n + 1))


def generate_base(
        pipe,
        dataset_root = './B',
        result_dir = './result',
        using_lora = True,
        best_step = None,
        model_step = 1200,
        output_num = 2,
        adain_queries = True,
        adain_keys = True,
        adain_values = False,
        use_shared_attention = False,
        attn_map_save_steps = [],
        use_inf_negative_prompt = True,
        change_args=None,
        ):
    model_version, need_inf_obj, need_inf_seed, need_ref_obj, taskid = change_args['model_version'], change_args['inf_obj'], change_args['seed'], change_args['ref_obj'], change_args['task_id']

    # seed for creating ref and inf init latent
    tar_seeds = create_number_list(output_num) 
    # tar_seeds = [tar_seeds[0]] + [tar_seeds[need_inf_seed]]
    # saving path creation
    if not using_lora:
        result_dir = f'{result_dir}_without_lora'
    elif using_lora:
        if best_step:
            result_dir = f'{result_dir}_best_step'
    if not os.path.exists(os.path.join(result_dir, taskid)):
        os.makedirs(os.path.join(result_dir, taskid))

    # load lora
    if using_lora:
        for lora_file in os.listdir(f"checkpoints/style{model_version}_{taskid}"):
            if best_step:
                if lora_file[:-4] == 'pytorch_lora_weights':
                    pipe.load_lora_weights(f"checkpoints/style{model_version}_{taskid}/{lora_file}")
            elif model_step is not None:
                if lora_file[:-4] != 'pytorch_lora_weights' and int(lora_file[:-4]) == model_step:
                    pipe.load_lora_weights(f"checkpoints/style{model_version}_{taskid}/{lora_file}")

    activate_layer_indices_list = [[[0,0],[18,24]]]
    activate_step_indices_list = [[[0,49]]]

    for activate_layer_indices in activate_layer_indices_list:
        for activate_step_indices in activate_step_indices_list:
            str_activate_layer, str_activate_step = pipe.activate_layer(activate_layer_indices=activate_layer_indices,
                                                                        attn_map_save_steps=attn_map_save_steps,
                                                                        activate_step_indices=activate_step_indices,
                                                                        use_shared_attention=use_shared_attention,
                                                                        adain_queries=adain_queries,
                                                                        adain_keys=adain_keys,
                                                                        adain_values=adain_values,)

    # prompt generation
    id2style = {'00':'{obj} of neon style','01':'an oil painting of {obj}','02':'{obj} of v02 style','03':'a drawing of {obj}','04':'{obj} of cloud style',
                '05':'{obj} of v05 style','06':'{obj} of chinese paper cutout style','07':'a drawing of {obj}','08':'{obj} of v08 style','09':'watercolor painting of {obj}',
                '10':'a drawing of {obj}','11':'{obj} of v11 style','12':'{obj} of lego bricks style','13':'{obj} of watercolor painting style','14':'{obj} of v14 style', 
                '15':'{obj} of v15 style','16':'{obj} in pixel art style','17':'a watercolor painting of {obj}', '18':'a drawing of {obj}', '19':'{obj} of lego bricks style',
                '20':'a drawing of {obj}','21':'a paper cut of {obj}', '22':'a drawing of {obj}', '23':'a drawing of {obj}','24':'{obj} of v24 style','25':'pixel art {obj}',
                '26':'{obj} of origami style','27':'{obj} in v27 style'}                                                                                     
    

    # ref propmt
    with open(f"{dataset_root}/{taskid}/train_prompt.json", "r") as file:
        ref_prompts = json.load(file)
    # inf obj
    with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
        inf_objs = json.load(file)                                                                    
    with jt.no_grad():
        
        # get corresponding ref img
        image_path = os.path.join(dataset_root,taskid,'images',f'{need_ref_obj}.png')
        real_img = Image.open(image_path).resize((512, 512), resample=Image.BICUBIC)
        latents = list()                   
        for tar_seed in tar_seeds: # 0 for ref latent, others for inf latent
            latents.append(init_latent_jt(model=pipe, seed=tar_seed))

        latents = jt.concat(latents, dim=0)
        latents.to('cuda')
        
        ref_prompt = ref_prompts[need_ref_obj] # from B_base/taskid/train_prompt.json

        inf_prompt = id2style[taskid].replace("{obj}", need_inf_obj.lower()) # f'{inf_obj} of {id2style[taskid]}' 
        images = pipe(  prompt = ref_prompt, 
                        guidance_scale = 7,
                        latents = latents,
                        num_images_per_prompt = len(tar_seeds),
                        target_prompt = inf_prompt, 
                        use_inf_negative_prompt = use_inf_negative_prompt,
                        num_inference_steps = 50,
                        image = real_img
                )[0]
        
        for i, img in enumerate(images):
            if i == need_inf_seed: # inference img
                save_path = os.path.join(result_dir, taskid, "{}.png".format(need_inf_obj))
                img.save(save_path)
                print(f"saved to {save_path}")


        # images_path = os.path.join(dataset_root,taskid,'images')
        # ref_objs = set()
        # for image_path in os.listdir(images_path):
        #     ref_obj = image_path[:-4]
        #     image_path = os.path.join(images_path,image_path)
        #     real_img = Image.open(image_path).resize((512, 512), resample=Image.BICUBIC)

        #     latents = list()                         
        #     for tar_seed in tar_seeds: 
        #         latents.append(init_latent_jt(model=pipe, seed=tar_seed))

        #     latents = jt.concat(latents, dim=0)
        #     latents.to('cuda')
            
        #     # ref_prompt = f'{ref_obj} of {id2style[taskid]}'
        #     ref_prompt = ref_prompts[ref_obj]
        #     # jump symmetrical image
        #     ref_obj = ref_obj.split('-')[-1]
        #     if ref_obj in ref_objs: 
        #         continue
        #     ref_objs.add(ref_obj)
        #     if ref_obj == need_ref_obj:
        #         for _, inf_obj in inf_objs.items():
        #             if inf_obj == need_inf_obj:
        #                 inf_prompt = id2style[taskid].replace("{obj}", inf_obj.lower())
        #                 images = pipe(  prompt = ref_prompt, 
        #                                 guidance_scale = 7,
        #                                 latents = latents,
        #                                 num_images_per_prompt = len(tar_seeds),
        #                                 target_prompt = inf_prompt, 
        #                                 use_inf_negative_prompt = use_inf_negative_prompt, # true
        #                                 num_inference_steps = 50,
        #                                 image = real_img
        #                     )[0]
                
        #                 for i, img in enumerate(images):
        #                      if i == need_inf_seed: # inference img
        #                         save_path = os.path.join(result_dir, taskid, "{}.png".format(need_inf_obj))
        #                         img.save(save_path)
        #                         print(f"saved to {save_path}")


    
