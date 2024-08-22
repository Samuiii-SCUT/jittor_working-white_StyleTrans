import json, os
os.environ['disable_lock'] = '1'
from JDiffusion.pipelines import StableDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from run_test import generate
from run_test_base import generate_base

def load_json(json_path):
    with open(json_path) as f:
        result=json.load(f)
    return result

id2args = load_json('./ljh_args.json')

# loading base model
color_cal_start_t, color_cal_window_size = 150, 50
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.fix_traj_t_start = color_cal_start_t
pipe.scheduler.fix_traj_t_end = color_cal_start_t - color_cal_window_size

def generate_ljh():
    for task_id, args in id2args.items():
        for single_args in args:
            if single_args is not None: 
                if single_args['scale_1']==0:
                    print(single_args)
                    generate_base(pipe,dataset_root='./B',result_dir='./result',change_args=single_args)
                elif single_args['scale_1']!=0:
                    print(single_args)
                    generate(pipe,dataset_root='./B2_v2',result_dir='./result',change_args=single_args)




