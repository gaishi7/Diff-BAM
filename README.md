# Diff-BAM: A Generalized Adaptive Diffusion Model for Cultural Heritage Image Inpaintinglatent-diffusion-inpainting
This study makes the following contributions：

Significantly reducing the computational overhead and inference delay of diffusion model by dynamically adjusting the computational path of denoising process.

The Gaussian fuzzy fusion module is embedded in U-Net to solve the complex edge transition problem of cultural heritage images.

We created a large-scale dataset of Thang_ka.
## Dataset
The core data for this study comes from our carefully constructed large-scale, high-quality Thang_ka image dataset, download link below:
[https://huggingface.co/datasets/zheng1/Thang_ka]


#### 1. Train

```
CUDA_VISIBLE_DEVICES=0 python main.py --base ldm/models/first_stage_models/vq-f4-noattn/config.yaml --resume ldm/models/first_stage_models/vq-f4-noattn/model.ckpt --stage 0 -t --gpus 0,

```

#### 2. Load and Inference
Please refer to those inference notebook.
python inpaint.py --indir Tang_ga/test --outdir outputs/result
