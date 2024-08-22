import json, os, shutil

os.makedirs('./checkpoints', exist_ok=True)
with open('./loras.json', 'r') as f:
    loras_dict = json.load(f)

weight_prefix = "/root/autodl-tmp/ljh/JDiffusion/examples/dreambooth/style/styleB"

for version in loras_dict.keys():
    for taskid in loras_dict[version].keys():
        for lora in loras_dict[version][taskid]:
            source_path = f"{weight_prefix}_{version}_{taskid}/{lora}.bin"
            os.makedirs(f'./checkpoints/styleB_{version}_{taskid}', exist_ok=True)
            shutil.copy(source_path, os.path.join(f'./checkpoints/styleB_{version}_{taskid}', os.path.basename(source_path)))