# DreamBooth-Lora

本项目参考自 [HuggingFace的训练指导](https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora) 。

## 环境安装

首先按照 [JDiffusion 的安装指导](../../README.md)安装必要的依赖，除此之外还需要安装 `peft` 库依赖。
```
pip install peft==0.10.0
```
## 训练

1. 首先从比赛云盘下载对应的数据集；
2. 设置训练所得到的模型版本，其中 B2-v2 对应版本 v2，B2 对应模型版本 v1；
3. 然后顺序运行 `python train.py --model_version B2` 以及 `python train.py --model_version B2_v2` 进行训练。模型的保存路径默认在checkpoints下

## 推理

1. 将 `run_all.py` 中的 `dataset_root` 修改为数据集对应的目录，将 `max_num` 修改为数据集中的风格个数；
2. 运行 `python run_all.py` 进行训练，对应的图片会输出到 `output` 文件夹。


## 参考文献

```
@inproceedings{ruiz2023dreambooth,
  title={Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
