## DDDM-EVC: Decoupled Denoising Diffusion Models for Emotional Voice Conversion

Implementation of the Decoupled Denoising Diffusion model for emotional voice conversion in the context of the master
thesis "Decoupled Denoising Diffusion for Emotional Voice Conversion".

Find the overview of the proposed model [here](https://wiki.alexanderbaur.de/).


## Setup

Install dependencies:

```bash
pip install poetry
poetry install  # for inference and training without visualization
poetry install --with visualize  # to visualize the training process
poetry install --with lint,test  # for development
```

Run tests:

```bash
poetry run pytest
```

Configuration uses [Hydra](https://hydra.cc/docs/intro/) and is located in `./config`.

## Training

Run the training:

```bash
poetry run python src/train.py
```

To start training from a specific checkpoint:

```bash
poetry run python src/train.py training.checkpoint=/path/to/ckpt.pth
poetry run python src/train.py training.checkpoint=latest
```

Setting to ``latest`` finds and loads the last checkpoint file in outputs directory.
Training produces outputs in the root of this project:

```
outputs/
├── 2025-02-16 /
│   ├── 13-56-17 /
│   │   ├── .hydra/                 # contains config used for run
│   │   ├── ckpt/                   # contains model checkpoints (`ckpt_e{epoch}_b{batch}.pth`)
│   │   ├── tensorboard/            # contains tensorboard data
│   │   ├── train.log               # contains logs of run
│   ├── ...
├── ...
```

Use `tensorboard --logdir ./outputs/2025-02-16/13-56-17/tensorboard` to visualize progress of a single training run.

## Inference

For voice conversion inference, run:

```
poetry run python src/inference.py [-h] [-s SOURCE_PATH] [-t TARGET_PATH] [-o OUTPUT_PATH] [-c CONFIG] [-n N_TIMESTEPS]

options:
  -s SOURCE_PATH, --source_path SOURCE_PATH
                        Path to the input audio wav
                        Default: ./sample/trg.wav
  -t TARGET_PATH, --target_path TARGET_PATH
                        Path to the conversion target audio wav
                        Default: ./sample/src.wav
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Dir for output wav
                        Default: ./sample/
  -c CONFIG, --config CONFIG
                        Name of config to use
                        Default: config_vc.yaml
  -n N_TIMESTEPS, --n_timesteps N_TIMESTEPS
                        Number of diffusion timesteps
                        Default: 6
```
