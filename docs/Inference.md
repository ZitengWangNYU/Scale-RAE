# Inference Guide

Complete guide for running Scale-RAE inference on GPU.

---

## Installation

```bash
cd Scale-RAE
pip install -e .
```

See main [README.md](../README.md) for initial setup.

---

## CLI Reference

### Text-to-Image Generation

```bash
python cli.py t2i \
  --prompt <text> \
  [--model-path <hf_repo>] \
  [--decoder-repo <hf_repo>] \
  [--output-dir <path>] \
  [--guidance-level <float>] \
  [--seed <int>] \
  [--save-latent]
```

**Arguments:**
- `--prompt`: Text description (required)
- `--model-path`: HF model repo (default: `nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B`)
- `--decoder-repo`: HF decoder repo (default: `nyu-visionx/siglip2_decoder`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--guidance-level`: Classifier-free guidance scale (default: 1.0)
- `--save-latent`: Save latent tensor for later reuse
- `--output-dir`: Output directory (default: `./outputs`)

**Examples:**

```bash
# Basic generation (uses SigLIP-2 model by default)
python cli.py t2i --prompt "Can you generate a photo of a cat on a windowsill?"

# Use WebSSL model with WebSSL decoder
python cli.py t2i \
  --model-path "nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B-WebSSL" \
  --decoder-repo "nyu-visionx/webssl300m_decoder" \
  --prompt "Can you create an image of a peaceful garden?"

# Save latent for later processing
python cli.py t2i \
  --prompt "Show me a photo of a serene lake at dawn" \
  --save-latent \
  --output-dir ./my_outputs

# Different seed for variation
python cli.py t2i \
  --prompt "Can you give me a photo of a futuristic robot?" \
  --seed 123
```

**Prompt Tips:**
- Use natural language: "Can you...", "I want...", "Show me..."
- Be descriptive but concise
- The model was trained on request-style prompts

---

### Image Understanding

```bash
python cli.py img \
  --image <path> \
  --prompt <text> \
  [--model-path <hf_repo>]
```

**Example:**

```bash
python cli.py img \
  --image "photo.jpg" \
  --prompt "Can you describe what's in this image?"
```

---

### Latent Operations

Work with saved latent tensors:

```bash
python cli.py latent \
  --latent <path> \
  --action {decode,qa,continue} \
  [--prompt <text>] \
  [--decoder-repo <hf_repo>]
```

**Actions:**
- `decode`: Convert latent to image
- `qa`: Answer questions about the latent
- `continue`: Continue generation from latent

**Examples:**

```bash
# Decode saved latent
python cli.py latent \
  --latent ./outputs/sample_123_latent.pt \
  --action decode

# Ask questions
python cli.py latent \
  --latent ./outputs/sample_123_latent.pt \
  --action qa \
  --prompt "What's in this image?"
```

---

## Scaling Experiments

Generate multiple samples per prompt and select the best using test-time compute scaling.

### Setup

```bash
pip install image-reward
```

### Basic Usage

```bash
python scaling_experiment.py t2i \
  --metadata-file example_prompts.jsonl \
  --output-dir ./scaling_outputs \
  --n-samples 4 \
  --scaling-rounds 2 \
  --post-qa-template-path ./assets/qa_prompt_template.txt \
  --post-qa-mode latent
```

### Key Parameters

**Generation:**
- `--n-samples`: Images to keep (e.g., 4)
- `--scaling-rounds`: Number of rounds (e.g., 2)
- **Total generated**: `n_samples × scaling_rounds` (e.g., 4 × 2 = 8)
- **Output**: Top `n_samples` selected by ImageReward + LLM scores

**Evaluation:**
- `--post-qa-mode`: Evaluation mode
  - `latent`: Faster, evaluates latent embeddings directly ⚡
  - `image`: Slower, evaluates decoded RGB images 🎯
- `--post-qa-template-path`: QA prompt template (default: `./assets/qa_prompt_template.txt`)

**Models:**
- `--model-path`: HF model repo
- `--decoder-repo`: HF decoder repo

### Prompt File Format

Create `example_prompts.jsonl`:

```json
{"index": 0, "prompt": "a photo of a horse and a giraffe"}
{"index": 1, "prompt": "a photo of a cake and a zebra"}
{"index": 2, "prompt": "a photo of a bottle and a refrigerator"}
```

**Note:** Prompts are automatically prefixed with "Can you generate" to match training format.

### Multi-GPU Usage

Automatically distributes prompts across GPUs:

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python scaling_experiment.py t2i \
  --metadata-file example_prompts.jsonl \
  --output-dir ./scaling_outputs \
  --n-samples 4 --scaling-rounds 2 \
  --post-qa-mode latent
```

**Distribution:** Prompts are split using `[rank::world_size]` pattern.

### Output Structure

```
scaling_outputs/
└── Scale-RAE-Qwen1-5B_DiT2-4B/           ← Model name (dots→dashes)
    └── 2_qa_prompt_template_scale_latent/ ← rounds_template_scale_mode
        ├── 00000/                         ← Prompt index 0
        │   ├── samples/
        │   │   ├── 00000.png              ← Generated images
        │   │   ├── 00001.png
        │   │   ├── ...
        │   │   └── 00007.png              ← (8 total for n=4, rounds=2)
        │   └── metadata.jsonl             ← Evaluation results
        ├── 00001/                         ← Prompt index 1
        └── 00002/                         ← Prompt index 2
```

### Understanding Results

**The metadata file contains:**

1. **ImageReward Scores** (aesthetic quality):
   ```json
   "all_ir_scores": [-0.614, 0.995, ..., 0.692],
   "top_ir_indices": [6, 4, 1, 5]  // Best 4 images by IR
   ```

2. **LLM Verification** (prompt alignment):
   ```json
   "text_responses": {
     "6": {
       "answer": "yes",           // Does image match prompt?
       "yes_conf_score": 8.41,    // Confidence (higher = better)
       "prompt_loss": 1.45,       // Lower = better match
       ...
     }
   }
   ```

**Selection Strategy:**
1. ImageReward ranks all images by aesthetic quality
2. LLM verifies top candidates for prompt alignment
3. Best image = highest ImageReward + LLM confirms "yes"

### Select Best Results

```bash
python verify_results.py \
  --sample-dir ./scaling_outputs/Scale-RAE-Qwen1-5B_DiT2-4B/2_qa_prompt_template_scale_latent \
  --metric combined
```

**Available metrics:**
- `combined`: Weighted combination of IR + LLM confidence (recommended)
- `answer_conf`: Select by LLM confidence score
- `image_reward`: Select by ImageReward score only

---

## How Scaling Works

**Process:**
1. Generate `n_samples × scaling_rounds` images (e.g., 4 × 2 = 8)
2. For each image: LLM self-evaluates ("Does this match?")
3. After all images: ImageReward scores aesthetic quality
4. Select top `n_samples` by ImageReward score
5. Verify with LLM confidence scores

**Why it works:** More samples → better chance of high-quality outputs → smart selection

**Cost vs Quality:**

| Config | Total Images | Quality Gain |
|--------|--------------|--------------|
| 4×1    | 4            | Baseline     |
| 4×2    | 8            | +15%         |
| 4×3    | 12           | +18%         |

**Sweet spot:** `n_samples=4, scaling_rounds=2`

---

## Output Files

**CLI outputs** (`./outputs/`):
- `sample_<timestamp>.png` - Generated image
- `sample_<timestamp>_manifest.json` - Metadata
- `sample_<timestamp>_latent.pt` - Latent tensor (if `--save-latent`)

**Scaling outputs** (`./scaling_outputs/`):
- `00000.png, 00001.png, ...` - All generated images
- `metadata.jsonl` - Complete evaluation results

---

## Advanced Topics

### Using Local Checkpoints

```bash
python cli.py t2i \
  --model-path /path/to/local/checkpoint \
  --decoder-repo /path/to/local/decoder \
  --prompt "Your prompt"
```

### Adjusting Guidance Scale

```bash
# Lower guidance (more creative, less faithful)
python cli.py t2i --prompt "A surreal landscape" --guidance-level 0.5

# Higher guidance (more faithful, less creative)
python cli.py t2i --prompt "A red apple" --guidance-level 1.5
```

### Batch Processing

Create a script to process multiple prompts:

```bash
#!/bin/bash
while IFS= read -r prompt; do
  python cli.py t2i --prompt "$prompt" --output-dir ./batch_outputs
done < prompts.txt
```

---

## Troubleshooting

**Out of Memory:**
- Use smaller model: `Scale-RAE-Qwen1.5B_DiT2.4B`
- Reduce `n_samples` or `scaling_rounds`
- Use `CUDA_VISIBLE_DEVICES` to select specific GPU

**Model Download Issues:**
- Models auto-download from HuggingFace
- Cached in `~/.cache/huggingface/`
- Check internet connection and HF credentials

**Poor Generation Quality:**
- Try different seeds: `--seed 42, 123, 456`
- Adjust guidance: `--guidance-level 0.8-1.2`
- Improve prompt: Be more specific and descriptive

---

## Citation

```bibtex
@article{scale-rae-2026,
  title={Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders},
  author={Shengbang Tong and Boyang Zheng and Ziteng Wang and Bingda Tang and Nanye Ma and Ellis Brown and Jihan Yang and Rob Fergus and Yann LeCun and Saining Xie},
  journal={arXiv preprint arXiv:2601.16208},
  year={2026}
}
```
