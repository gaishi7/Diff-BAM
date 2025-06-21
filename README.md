# Diff-BAM: A Generalized Adaptive Diffusion Model for Cultural Heritage Image Inpaintinglatent-diffusion-inpainting

This repository is based on [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion), with modifications for classifier conditioning and architecture improvements.

通过动态调整去噪过程的计算路径，显著降低扩散模型的计算开销与推理延迟.

在U-Net中嵌入高斯模糊融合模块，解决文化遗产图像复杂的边缘过渡问题.

## Dataset
本研究的核心数据来源于我们精心构建的大规模高质量唐卡图像数据集.部分图片如下图所示：


#### 1. Train

```
CUDA_VISIBLE_DEVICES=0 python main.py --base ldm/models/first_stage_models/vq-f4-noattn/config.yaml --resume ldm/models/first_stage_models/vq-f4-noattn/model.ckpt --stage 0 -t --gpus 0,

```
#### 2. Comebine the autoencoder with the diffusion model
Please refer to the combine.ipynb
#### 3. Finetune Latent diffusion model

Note that, the mask in here is in square mask, you can disable draw_rectangle_over_mask function in the /ldm/ldm/data/PIL_data.py to use original mask.

![rdm-figure](assets/abc.png)



First, download the pre trained weight and prepare the images with the same format as in kvasir-seg folder

Download the pre-trained weights
```
wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

Second, modify the data path in config.yaml( it should be in ldm/models/ldm/inpainting_big/config.yaml )

Then, run the following command
```
CUDA_VISIBLE_DEVICES=0 python main.py --base ldm/models/ldm/inpainting_big/config.yaml --resume ldm/models/ldm/inpainting_big/last.ckpt --stage 1 -t --gpus 0,
pytorch_lightning==1.6.5
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
先将生成的auto和下载的权重结合，然后添加代码到main中
        model_path = 'celeba_auto.ckpt'
        config.model['params']['ckpt_path'] = model_path
        model = instantiate_from_config(config.model)
最后执行下面代码
CUDA_VISIBLE_DEVICES=0 python main.py --base ldm/models/ldm/inpainting_big/config.yaml --stage 1 -t --gpus 0,
```

#### 4. Load and Inference
Please refer to those inference notebook.
python inpaint.py --indir Tang_ga/test --outdir outputs/result
