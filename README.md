## Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders (Scale RAE) <br><sub>Official Implementation</sub>

### [Paper](https://arxiv.org/abs/2601.16208) | [Project Page](https://rae-dit.github.io/scale-rae/) | [Models](https://huggingface.co/collections/nyu-visionx/scale-rae) | [Data](https://huggingface.co/datasets/nyu-visionx/scale-rae-data)

This repository provides **GPU inference** and **TPU training** implementations for our paper: 
Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders.

> [**Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders**](https://rae-dit.github.io/scale-rae/)<br>
> [Shengbang Tong](https://tsb0601.github.io/)\*, [Boyang Zheng](https://bytetriper.github.io/)\*, [Ziteng Wang](https://github.com/ZitengWangNYU)\*, [Bingda Tang](https://tang-bd.github.io/), [Nanye Ma](https://willisma.github.io), [Ellis Brown](https://ellisbrown.github.io/), [Jihan Yang](https://jihanyang.github.io/), [Rob Fergus](https://www.cs.nyu.edu/~fergus/pmwiki/pmwiki.php), [Yann LeCun](http://yann.lecun.com/), [Saining Xie](https://www.sainingxie.com)
> <br>New York University<br>
> <sup>*</sup>Core contributor

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ZitengWangNYU/Scale-RAE.git
cd Scale-RAE
conda create -n scale_rae python=3.10 -y
conda activate scale_rae
pip install -e .
```

### Inference

```bash
cd inference
python cli.py t2i --prompt "Can you generate a photo of a cat on a windowsill?"
```

Models and decoders automatically download from HuggingFace.

---

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| **[Inference Guide](docs/Inference.md)** | Generate images with pre-trained models |
| **[Training Guide](docs/Train.md)** | Train your own Scale-RAE models |
| **[TPU Setup Guide](docs/TPUs_Torch_XLA.md)** | Set up TPUs for large-scale training |

---

## 📦 Available Models

All models available in our [HuggingFace collection](https://huggingface.co/collections/nyu-visionx/scale-rae):

| Model | LLM | DiT | Decoder | HuggingFace Repo |
|-------|-----|-----|---------|------------------|
| **Scale-RAE** | Qwen2.5-1.5B | 2.4B | SigLIP-2 | `nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B` ⭐ |
| **Scale-RAE** | Qwen2.5-7B | 9.8B | SigLIP-2 | `nyu-visionx/Scale-RAE-Qwen7B_DiT9.8B` |
| **Scale-RAE-WebSSL** | Qwen2.5-1.5B | 2.4B | WebSSL | `nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL` |

⭐ = Recommended default model

**Decoders:**
- `nyu-visionx/siglip2_decoder` (SigLIP-2-SO400M, default)
- `nyu-visionx/webssl300m_decoder` (WebSSL-DINO300M)

---

## 🎓 Training

Scale-RAE follows a two-stage training approach:

1. **Stage 1**: Large-scale pretraining with pretrained LLM and randomly initialized DiTs
2. **Stage 2**: Finetuning on high-quality instruction datasets

### Example Scripts

```bash
# Stage 1: Pretraining with SigLIP-2
bash scripts/examples/stage1_rae_siglip_1.5b_dit2.4b.sh

# Stage 2: Instruction finetuning
bash scripts/examples/stage2_rae_siglip_1.5b_dit2.4b.sh
```

See [Training Guide](docs/Train.md) for data preparation, hyperparameters, and example scripts. See [TPU Setup Guide](docs/TPUs_Torch_XLA.md) for TPU configuration.

---

## 🏗️ Repository Structure

```
Scale-RAE/
├── inference/              # Inference CLI and scaling experiments
├── scale_rae/             # Core model implementation
│   ├── model/             # Model architectures (LLM, DiT, encoders)
│   └── train/             # Training scripts (SPMD/FSDP)
├── scripts/examples/      # Example training scripts
├── docs/
│   ├── Inference.md       # Inference guide (CLI, scaling)
│   ├── Train.md           # Training guide (data, hyperparams)
│   └── TPUs_Torch_XLA.md  # TPU setup guide
├── setup_gcs_mount.sh     # GCS mount for WebDataset
├── install_spmd.sh        # TPU/TorchXLA installation
└── clear.py               # TPU memory clearing utility
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{scale-rae-2026,
  title={Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders},
  author={Shengbang Tong and Boyang Zheng and Ziteng Wang and Bingda Tang and Nanye Ma and Ellis Brown and Jihan Yang and Rob Fergus and Yann LeCun and Saining Xie},
  journal={arXiv preprint arXiv:2601.16208},
  year={2026}
}
```

---

## 📄 License

This project is released under the MIT License.

---

## 🙏 Acknowledgments

This work builds upon:
- [RAE](https://github.com/bytetriper/RAE) - Diffusion Transformers with Representation Autoencoders
- [Cambrian-1](https://github.com/cambrian-mllm/cambrian) - Multimodal LLM framework
- [WebSSL](https://github.com/facebookresearch/webssl) - Self-supervised vision models
- [SigLIP-2](https://github.com/google-research/big_vision) - Self-supervised & language-supervised vision models