# Scale-RAE Training Guide

## Training Overview

Scale-RAE training follows a two-stage approach:
1. **Stage 1: Large-scale pretraining** with pretrained LLM and randomly initialized DiT
2. **Stage 2: Fine-tuning** on smaller high-quality datasets

Both stages use frozen vision encoder/decoder (RAE components). The decoder is only needed for inference and not loaded during training.

---

## Key Training Parameters

### Model Architecture

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--model_name_or_path` | LLM or checkpoint | `Qwen/Qwen2.5-1.5B-Instruct` (Stage 1)<br>Stage 1 checkpoint (Stage 2) |
| `--diffusion_model_depth` | Total DiT layers | `32` (2.4B), `48` (9.8B) |
| `--diffusion_model_heads` | Attention heads | `32` (2.4B), `48` (9.8B) |
| `--diffusion_model_hidden_size` | Hidden dimension | `2048` (2.4B), `3072` (9.8B) |
| `--diffusion_model_z_channels` | Conditioning dimension | `2048` (2.4B), `3072` (9.8B) |

### Vision Encoder (Frozen)

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--vision_tower_aux_list` | Vision encoder (frozen) | `["google/siglip2-so400m-patch14-224"]`<br>`["facebook/webssl-dino300m-full2b-224"]` |
| `--unfreeze_mm_vision_tower` | Always frozen | `False` |

### VAE Alignment (Optional)

| Parameter | Description | Usage |
|-----------|-------------|-------|
| `--generation_alignment_tower` | VAE encoder for target | `black-forest-labs/FLUX.1-dev-res256` |
| `--diffusion_model_channels` | DiT input channels | `64` (for VAE latents) |

### Optimization

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--learning_rate` | LLM learning rate | `5.65e-5` |
| `--diff_head_lr` | DiT learning rate | `5.65e-4` |
| `--warmup_ratio` | Warmup ratio | `0.0134` (Stage 1), `0.03` (Stage 2) |
| `--per_device_train_batch_size` | Batch size per device | `128` |
| `--num_train_epochs` | Training epochs | `1` (Stage 1), `4` (Stage 2) |
| `--model_max_length` | Sequence length | `512` (Stage 1), `1024` (Stage 2) |

---

## Training Data

### Data Sources

Our training uses the following publicly available datasets:

| Dataset | Source 🤗 | Usage |
|---------|--------|-------|
| **BLIP-3o Web Images** (~39M) | [BLIP3o/datasets](https://huggingface.co/BLIP3o/datasets) | Decoder & Stage 1 Pretraining |
| **FLUX Synthetic Data** (~24.7M) | [nyu-visionx/scale-rae-data](https://huggingface.co/datasets/nyu-visionx/scale-rae-data) | Decoder & Stage 1 Pretraining |
| **RenderedText** (first 10M) | [wendlerc/RenderedText](https://huggingface.co/datasets/wendlerc/RenderedText) | Decoder only |
| **Cambrian-7M** | [nyu-visionx/Cambrian-10M](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M) | Stage 1 Pretraining only |
| **BLIP-3o-60k** | [BLIP3o/datasets](https://huggingface.co/BLIP3o/datasets) | Stage 2 Finetuning only |

### Training Pipeline

**RAE Decoder Training (~73M images):**
- Web + Synthetic + Text-rendering data
- Trains vision encoder/decoder only

**Unified Model Stage 1 - Pretraining (~70M images):**
- Web + Synthetic + Cambrian-7M
- Format: WebDataset (TAR files with `.jpg` + `.json` pairs)
- Trains LLM + DiT with frozen encoder/decoder

**Unified Model Stage 2 - Finetuning (60k images):**
- BLIP-3o-60k high-quality instructions
- Format: JSONL (`{"image": "path.jpg", "conversations": [...]}`)
- Fine-tunes LLM + DiT

### Data Format

**JSONL Format:**
```jsonl
{"image": "/path/to/image.jpg", "conversations": [{"from": "human", "value": "Generate a sunset"}, {"from": "gpt", "value": "Here you go:\n<image>"}], "metadata": {}}
```

**WebDataset Format:**
Tar archives with paired `sample_XXXXXX.json` + `sample_XXXXXX.png` files.

Manifest (`wds_manifest.json`):
```json
{
  "dataset_name": "my_training_data",
  "tars": [
    "/path/to/dataset_batch_001.tar",
    "/path/to/dataset_batch_002.tar"
  ]
}
```

Sample JSON inside tar (`sample_000001.json`):
```json
{
  "conversations": [
    {"from": "human", "value": "Generate a picture of a sunset over mountains"},
    {"from": "gpt", "value": "Here you go:\n<image>"}
  ],
  "metadata": {},
  "image_filename": "sample_000001.png"
}
```

---

## Quick Start

For detailed TPU setup, see [TPUs_Torch_XLA.md](TPUs_Torch_XLA.md).

### 1. Prepare Data
Set paths in training script:
```bash
DATA_PATH="/path/to/dataset.jsonl"  # or wds_manifest.json
IMAGE_FOLDER=""  # Empty for JSONL
```

### 2. Run Training

**Stage 1: Large-scale Pretraining**

RAE-SigLIP (1.5B + 2.4B DiT):
```bash
bash scripts/examples/stage1_rae_siglip_1.5b_dit2.4b.sh
```

RAE-WebSSL (1.5B + 2.4B DiT):
```bash
bash scripts/examples/stage1_rae_webssl_1.5b_dit2.4b.sh
```

VAE Alignment (1.5B + 2.4B DiT):
```bash
bash scripts/examples/stage1_vae_alignment_1.5b_dit2.4b.sh
```

**Stage 2: Fine-tuning**

Fine-tune Stage 1 checkpoint:
```bash
bash scripts/examples/stage2_rae_siglip_1.5b_dit2.4b.sh
```
- Load Stage 1 checkpoint via `--model_name_or_path`
- Train on high-quality instruction data (BLIP-3o-60k)
- 4 epochs with increased warmup ratio

---

## Additional Resources

- **Models:** https://huggingface.co/collections/nyu-visionx/scale-rae
- **Paper:** https://arxiv.org/abs/2601.16208
- **Inference:** [inference/README.md](../inference/README.md)
