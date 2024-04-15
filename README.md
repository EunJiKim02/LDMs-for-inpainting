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
pip install blobfile
pip install easydict
```

### Dataset 준비

1. mask 생성

```python
#python scripts/get_mask_dataset.py --config [GEN_MASK_CONFIG] --indir [INPUT IMAGE DIR] --outdir [OUTPUT DIR]

python scripts/gen_mask_dataset.py --config scripts/gen_mask_config/random_medium.yaml --indir data/stanford_dog/train/ --outdir data/stanford_dog_mask/train/ 
python scripts/gen_mask_dataset.py --config scripts/gen_mask_config/random_medium.yaml --indir data/stanford_dog/val/ --outdir data/stanford_dog_mask/val/ 
```

2. csv 생성

```python
# python scripts/generate_csv.py --llama_masked_outdir [MASK DIR] --csv_out_path [CSV PATH(.csv)]
python scripts/generate_csv.py --llama_masked_outdir data/stanford_dog_mask/train --csv_out_path data/stanford_dog_mask/stanford_dog_mask_train.csv
python scripts/generate_csv.py --llama_masked_outdir data/stanford_dog_mask/val --csv_out_path data/stanford_dog_mask/stanford_dog_mask_val.csv
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
