import json, os, tqdm
import jittor as jt
from utils import init_latent_jt
from PIL import Image
import logging

# save params
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # Output to file
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Output to terminal
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def create_number_list(n):
    return list(range(n))


def generate(
        pipe, 
        dataset_root='./B2_v2',
        result_dir='./result',
        using_lora=True,
        best_step=None,
        saving_log=False,
        output_num=3,
        model_step=1200,
        num_inference_steps=50,
        activate_step_indices = [[0,49]],
        content_activate_step_indices = [[0,49]], # for content
        activate_key_step = 30,
        scale_in_cross = False,
        adain_queries = False,
        use_shared_attention = False,
        attn_map_save_steps = [],
        adain_keys = True,
        adain_values = False,
        change_args=None,
        ):
    scale_1, scale_2, activate_layer_indices, content_activate_layer_indices, taskid, model_version, need_inf_obj = change_args['scale_1'], change_args['scale_2'], change_args['activate_layer_indices'], change_args['content_activate_layer_indices'], change_args['task_id'], change_args['model_version'], change_args['inf_obj']

    # seed for creating ref and inf init latent
    tar_seeds = create_number_list(output_num) 

    # saving path creation
    if not using_lora:
        result_dir = f'{result_dir}_without_lora'
    elif using_lora:
        if best_step:
            result_dir = f'{result_dir}_best_step'
    if not os.path.exists(os.path.join(result_dir, taskid)):
        os.makedirs(os.path.join(result_dir, taskid))
    save_path = os.path.join(result_dir, taskid)

    if saving_log:
        log_save_path = os.path.join(result_dir, f'./params_{taskid}.log')
        logger = get_logger(log_save_path)
        logger.info('save_path={}'.format(save_path))
        logger.info('model_step={}\t model_version={}\t scale_1={}\t sacle_2={}\t activater_layer_indices={}\t activate_step_indices={}\t content_activate_layer_indices={}\t content_activate_step_indices={}\t activate_key_step={}'.format(model_step, model_version, scale_1, scale_2, activate_layer_indices, activate_step_indices, content_activate_layer_indices, content_activate_step_indices, activate_key_step))

    # load lora
    if using_lora:
        for lora_file in os.listdir(f"style/style{model_version}_{taskid}"):
            if best_step:
                if lora_file[:-4] == 'pytorch_lora_weights':
                    pipe.load_lora_weights(f"style/style{model_version}_{taskid}/{lora_file}")
            elif model_step is not None:
                if lora_file[:-4] != 'pytorch_lora_weights' and lora_file[:-4] != 'lowest_loss' and int(lora_file[:-4]) == model_step:
                    pipe.load_lora_weights(f"style/style{model_version}_{taskid}/{lora_file}")

    # loading model and replace with cross-attention
    str_activate_layer, str_activate_step = pipe.activate_layer(activate_layer_indices=activate_layer_indices,
                                                            content_activate_layer_indices=content_activate_layer_indices,
                                                            attn_map_save_steps=attn_map_save_steps,
                                                            activate_step_indices=activate_step_indices,
                                                            content_activate_step_indices=content_activate_step_indices,
                                                            use_shared_attention=use_shared_attention,
                                                            adain_queries=adain_queries,
                                                            adain_keys=adain_keys,
                                                            adain_values=adain_values,
                                                            style_id=True,
                                                            scale_1=scale_1,
                                                            scale_2=scale_2,
                                                            scale_in_cross=scale_in_cross
                                                                )


                                                                                    
    # style description dict
    id2style = {'00':'{obj} of neon style','01':'an oil painting of {obj}','02':'{obj} of v02 style','03':'a drawing of {obj}','04':'a/an {obj} in v04 style',
                '05':'{obj} of v05 style','06':'{obj} of chinese paper cutout style','07':'a drawing of {obj}','08':'{obj} of v08 style','09':'watercolor painting of {obj}',
                '10':'a drawing of {obj}','11':'{obj} of v11 style','12':'{obj} of lego bricks style','13':'{obj} of watercolor painting style','14':'{obj} of v14 style', 
                '15':'{obj} of v15 style','16':'{obj} in pixel art style','17':'a watercolor painting of {obj}', '18':'a drawing of {obj}', '19':'{obj} of lego bricks style',
                '20':'a drawing of {obj}','21':'a paper cut of {obj}', '22':'a drawing of {obj}', '23':'a drawing of {obj}','24':'{obj} of v24 style','25':'pixel art {obj}',
                '26':'{obj} of origami style','27':'{obj} in v27 style'}                                                                                     
    # inf obj all_case
    with open(f"./B2_v2/{taskid}/prompt.json", "r") as file:
        inf_objs = json.load(file) 
                                                                        
    with jt.no_grad():
        images_path = os.path.join(dataset_root,taskid,'images')
        ref_objs = set()
        for image_path in os.listdir(images_path):
            ref_obj = image_path[:-4]
            image_path = os.path.join(images_path,image_path)
            real_img = Image.open(image_path).resize((512, 512), resample=Image.BICUBIC)
            
            latents = list()                         
            for tar_seed in tar_seeds: # 0 for style latent, 1 for content latent, 2 for cs latent [s,c,cs,inf1,inf2,...]
                latents.append(init_latent_jt(model=pipe, seed=tar_seed))

            latents = jt.concat(latents, dim=0) # num_of_tar_seeds, latent_dim, h_scale, w_scale
            latents.to('cuda')
            
            # jump symmetrical image
            ref_obj = ref_obj.split('-')[-1]
            if ref_obj in ref_objs: 
                continue
            ref_objs.add(ref_obj)

            # only use one ref img to test
            if len(ref_objs) == 2:
                break

            for _, inf_obj in inf_objs.items():
                if inf_obj != need_inf_obj:
                    continue
                image_content_path = f"./B_ref/{taskid}/{inf_obj}.png"
                real_img_content = Image.open(image_content_path).resize((512, 512), resample=Image.BICUBIC)
        

                cs_mix_prompt = id2style[taskid].replace("{obj}", inf_obj.lower()) # f'{inf_obj} of {id2style[taskid]}' 
                c_prompt = f"a painting of {inf_obj.lower()}"
                r_prompt = id2style[taskid].replace("{obj}", ref_obj.lower())
                # checking
                print(r_prompt,c_prompt,cs_mix_prompt)
                # for input [s,c,cs], [ref_obj of cloud style, inf_obj, inf_obj of cloud style] 
                
                images = pipe(  prompt = r_prompt, 
                                guidance_scale = 7,
                                latents = latents,
                                num_images_per_prompt = len(tar_seeds), # The number of images to generate per prompt.
                                c_prompt = c_prompt,
                                cs_mix_prompt = cs_mix_prompt,
                                use_inf_negative_prompt = False,
                                use_advanced_sampling = False,
                                use_prompt_as_null = True,
                                num_inference_steps = num_inference_steps,
                                image = real_img,
                                image_content = real_img_content
                            )[0]
                for i, img in enumerate(images):
                    if i == 2: # inference img
                        img_save_path = os.path.join(save_path, "{}.png".format(inf_obj))
                        img.save(img_save_path)
                        print(f"saved to {img_save_path}")

    # reset to initial state
    str_activate_layer, str_activate_step = pipe.activate_layer(activate_layer_indices=[[0,0]],
                                                            content_activate_layer_indices=[[0,0]],
                                                            attn_map_save_steps=attn_map_save_steps,
                                                            activate_step_indices=activate_step_indices,
                                                            content_activate_step_indices=content_activate_step_indices,
                                                            use_shared_attention=use_shared_attention,
                                                            adain_queries=adain_queries,
                                                            adain_keys=adain_keys,
                                                            adain_values=adain_values,
                                                            style_id=True,
                                                            scale_1=scale_1,
                                                            scale_2=scale_2,
                                                            scale_in_cross=scale_in_cross
                                                                )
    

        


