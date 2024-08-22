# StyleTrans-第四届计图人工智能挑战赛-赛题二：风格迁移-008-打工小白-华南理工大学

![image](./assets/image.png)

## 简介

本项目包含了第四届计图挑战赛计图 - 风格迁移图片生成比赛的计图代码实现。基于逐步逐层风格，内容注意力注入方法在Dreambooth-lora的基础上进行改进，在赛题B榜中得分为0.4721，排名第8.

### 实现

StyleTrans 基于计图及其衍生套件实现： [Jittor](https://github.com/Jittor/jittor), [Jtorch](https://github.com/JITTorch/jtorch), [diffusers](https://github.com/huggingface/diffusers).

预训练模型包含：[stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)

### 简要思路

为了使得生成的图片在语义上与提示词更加对齐、在风格上与参考图片更为接近，我们修改了推理过程中UNet的自注意力模块、引入风格参考图像和内容参考图像，经过inversion后分别对自注意力计算中的查询向量，键值向量做向量注入，使得风格迁移的过程更加稳定。

其中风格参考图像从官方提供的参考图片中选取，内容参考图像由经风格18参考图片微调后的SD2.1模型生成。

### 测试环境

- Ubuntu 20.04.6 LTS
- jittor float32 推理需要30gb vram，在40gb vram环境中测试
- cuda 11.6

## 安装

### 0. conda环境准备

```bash
conda create -n jdiffusion python=3.9
conda activate jdiffusion
```

### 1. 安装依赖

```bash
pip install ./jittor
pip install ./jtorch
pip install ./diffusers_jittor
pip install ./transformers_jittor
pip install accelerate==0.27.2
pip install peft==0.10.0
pip install einops
```

或者

```bash
pip install -r requirement.txt
```

### 2. 安装JDiffusion

```bash
pip install -e .
```

### 3. 其他依赖

 If you encounter `No module named 'cupy'`:

```bash
# Install CuPy from source
pip install cupy
# Install CuPy for cuda11.2 (Recommended, change cuda version you use)
pip install cupy-cuda112
```

## 训练

## 推理
按照赛事要求，我们提供了test.py推理所有结果
请确保：
lora权重在：`examples/dreambooth/checkpoints`，子目录格式形如`style_v1_00`
数据在：`examples/dreambooth/B`，`examples/dreambooth/B2`，`examples/dreambooth/B2_v2`，其中后两者经过反转数据预处理和Blip获得训练描述

运行 test.py

```python
cd examples/dreambooth
python test.py
```