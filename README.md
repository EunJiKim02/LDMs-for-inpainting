# LDMs-for-inpainting

### 참고 자료

- Main

https://github.com/CompVis/latent-diffusion

- dataset prepare

https://github.com/advimman/lama.git

- Additional Inpainting source

https://github.com/CreamyLong/stable-diffusion.git

https://github.com/nickyisadog/latent-diffusion-inpainting

[Stable-Diffusion-Inpaint/ldm/models/diffusion/ddpm.py at 2ef9959c2eada05d34e3275d19fec8a54d449e9f · lorenzo-stacchio/Stable-Diffusion-Inpaint](https://github.com/lorenzo-stacchio/Stable-Diffusion-Inpaint/blob/2ef9959c2eada05d34e3275d19fec8a54d449e9f/ldm/models/diffusion/ddpm.py)

### 환경 세팅

- git clone

```python
git clone "https://github.com/EunJiKim02/LDMs-for-inpainting.git"
```

- conda environment

```python
# 아직 업데이트 중
conda env create -f enviroment.yaml
conda activate ldm
```

- 아직 update안한 library

```python
pip install blobfile==2.1.1
pip install easydict==1.13
```

### 기존 pre-trained model download
```python
wget -O ldm/origin.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

### Dataset 준비

1. mask 생성

```python
#python scripts/get_mask_dataset.py --config [GEN_MASK_CONFIG] --indir [INPUT IMAGE DIR] --outdir [OUTPUT DIR]
python scripts/gen_mask_dataset.py --config scripts/gen_mask_config/random_medium.yaml --indir data/churches_train/ --outdir data/churches_mask 
```

2. csv 생성

```python
# python scripts/generate_csv.py --llama_masked_outdir [MASK DIR] --csv_out_path [CSV PATH(.csv)]
python scripts/generate_csv.py --llama_masked_outdir data/churches_mask/ --csv_out_path data/churches_mask.csv
```

### 실행 방법 (Model Training)

- finetuning

```jsx
python main.py --base [config.yaml] -t --gpus 0,
```

### Inference

```jsx
python inpaint.py --indir [INPUT_DIR] --outdir [OUTPUT_DIR]
```
