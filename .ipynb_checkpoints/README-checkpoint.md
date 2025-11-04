# Catchment MAE — Scale-aware Masked Autoencoder for Catchment TIFFs

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
![Python-tested](https://img.shields.io/badge/python-3.8%2B-brightgreen)

**Catchment MAE** is a small toolkit + Flask UI to train a scale-aware Masked Autoencoder (MAE) on geospatial / catchment TIFF images. It supports single- or multi-channel TIFFs, includes a lightweight web UI to kick off training and request embeddings, and adds a simple scale/metadata embedding to make the MAE aware of original image size.

This repository contains:

* `app.py` — Flask application with training & inference endpoints (UI & API).
* `model/` — MAE-related modules (uses `timm` + Lightly-style masked ViT modules).
* Training loop, TIFF dataset loader, and a single-image embedding endpoint.

---

## Highlights

* Trains a Masked Autoencoder (MAE) with a small scale embedding network so the model sees the image’s physical scale (original H, W) when encoding.
* Supports TIFF inputs (single or multi-channel).
* Simple web UI for launching training and pulling embedding vectors from a trained `.pth`.
* Lightweight Python stack — PyTorch + timm + Lightly modules (used for masked ViT & MAE decoder modules).

---

## Quickstart

> The repository includes a small Flask UI to start training and fetch embeddings. The UI runs on `http://127.0.0.1:5000/`.

1. Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install torch torchvision timm flask tifffile numpy imagecodecs
# If you use Lightly modules from PyPI:
pip install lightly
```

3. Run the web app:

```bash
python app.py
```

4. Open your browser at `http://127.0.0.1:5000` and fill in the training form:

* Dataset directory: path containing `.tif` / `.tiff` files
* Output model path: `catchment_foundation_model.pth` (or your preferred path)
* Adjust ViT size, image size, mask ratio, epochs, batch size, etc.

---

## Usage (CLI / programmatic)

If you prefer not to use the UI, the training function is contained in `app.py` (`run_training_thread`) and expects a `config` dict of options:

```python
config = {
    "input_dir": "/path/to/tiffs",
    "model_output_path": "catchment_foundation_model.pth",
    "vit_model_name": "vit_small_patch16_224",
    "image_size": 224,
    "in_chans": 1,
    "norm_mean": [2000.0],
    "norm_std": [1000.0],
    "max_epochs": 50,
    "batch_size": 32,
    "mask_ratio": 0.75,
    "num_workers": 4,
}
# call run_training_thread(config) or start via Flask UI form
```

Saved outputs:

* `*.pth` — model weights
* `*.json` — config used to train (same name as `.pth`, `.json` extension)

---

## TIFF dataset format

* Supported file extensions: `.tif`, `.tiff`
* Single-channel TIFFs are supported (default). Multi-channel is supported if you set `in_chans` accordingly.
* The dataset loader preserves the original image shape in `metadata` (H, W) — this is used by the `scale_embed` network.

---

## Model notes

* Encoder: a ViT from `timm`, wrapped via MaskedVisionTransformerTIMM
* Decoder: MAEDecoderTIMM
* `scale_embed` is a small MLP that consumes `log(original_shape + eps)` and produces embeddings added to patch tokens
* Masking: random patch permutation + keep ratio determined by `mask_ratio`

Important helper utilities are included:

* `patchify(imgs, patch_size)` — to produce target patches for MAE reconstruction loss
* `collate_fn_skip_corrupt` — filters unreadable images in the dataloader

---

## Inference / Get embedding

Use the `/embed` endpoint (form provided in the web UI) or call programmatically:

1. Ensure you have `catchment_foundation_model.pth` and the corresponding `.json`.
2. POST `model_path` and `image_path` to `/embed`.
3. The endpoint returns:

```json
{
  "status": "success",
  "embedding": [ ... ],
  "shape": [D]
}
```

The returned embedding is the model’s CLS-token / global embedding: `encoded_tokens[:, 0]`.

---

## Dependencies

Minimum recommendations (matching what the project uses):

* Python 3.8+
* PyTorch >= 1.11 (or current stable for your GPU)
* torchvision
* timm
* tifffile
* imagecodecs (recommended if TIFFs are compressed)
* lightly (optional / acknowledged — see below)

Install:

```bash
pip install torch torchvision timm tifffile numpy imagecodecs flask
# optionally
pip install lightly
```

---

## Acknowledgements

This project uses ideas and components inspired by the Masked Autoencoder (MAE) literature and borrows implementation patterns from the open-source ecosystem. In particular, I want to **acknowledge the Lightly project**:

* **Lightly** ([https://github.com/lightly-ai/lightly](https://github.com/lightly-ai/lightly), [https://docs.lightly.ai](https://docs.lightly.ai)) provides a modular, well-documented self-supervised learning ecosystem with implementations and utilities for many SSL methods and masked ViT / MAE style modules. Portions of this project’s masking/VIT/MAE module usage and training patterns were adapted with inspiration from Lightly’s codebase and documentation. If you use this code in a publication or product, please consider citing or acknowledging Lightly and the original MAE papers where applicable.

Helpful Lightly resources:

* Docs: [https://docs.lightly.ai/self-supervised-learning/](https://docs.lightly.ai/self-supervised-learning/)
* Repo: [https://github.com/lightly-ai/lightly](https://github.com/lightly-ai/lightly)

Also inspired by:

* Masked Autoencoders (MAE) — He et al., 2021. (useful background reading)

---

## License

This repository is provided under the MIT License — see `LICENSE` (or add an appropriate license file). Third-party dependencies (Lightly, timm, PyTorch, etc.) remain under their original licenses.

---

[Back to top 🚀](#top)
