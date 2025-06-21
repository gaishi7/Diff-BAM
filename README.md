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

#### 2. Load and Inference
Please refer to those inference notebook.
python inpaint.py --indir Tang_ga/test --outdir outputs/result
