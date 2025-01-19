## DDDM-EVC: Decoupled Denoising Diffusion Models for Emotional Voice Conversion

Implementation of the Decoupled Denoising Diffusion model for emotional voice conversion in the context of the master
thesis "Decoupled Denoising Diffusion for Emotional Voice Conversion".

Find the overview of the proposed model [here](https://wiki.alexanderbaur.de/Master%20Thesis/Overview/).

Install dependencies:

```bash
pip install poetry
poetry install
```

Run the training:

```bash
poetry run python src/train.py
```

Run tests:

```bash
poetry run pytest
```
