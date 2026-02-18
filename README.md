# Orpheus TTS — Voice Clone

Fine-tuning [Orpheus 3B TTS](https://github.com/canopylabs/orpheus-tts) on a speaking voice using LoRA adapters.

## Overview

This project creates a custom TTS voice by fine-tuning the Orpheus 3B text-to-speech model on audio clips of a speaking voice extracted from interviews, behind-the-scenes footage, and commentary. The result is a LoRA adapter that can generate speech in her voice from arbitrary text.

## Training Data

### v3 Dataset (current)

| Metric | Value |
|--------|-------|
| Total clips | 511 |
| Total duration | 81.8 minutes |
| Train split | 459 clips (73.8 min) |
| Validation split | 52 clips (8.1 min) |
| Format | 24kHz mono WAV |
| Source | Interviews, commentary, behind-the-scenes footage |
| Clips with prosody tags | 260 (50.9%) |
| Total prosody tag instances | 329 |

### v3 Improvements over v1
- **Re-transcribed** with Whisper large-v3 (better punctuation, word accuracy, word-level timestamps)
- **Quality curated** — 80 low-quality clips removed (35 too quiet, 45 below quality threshold)
- **Prosody annotated** — Orpheus emotion tags (`<laugh>`, `<chuckle>`, `<sigh>`, `<gasp>`) inserted based on audio analysis
- See [PROSODY.md](PROSODY.md) for full annotation methodology

### Prosody Tag Distribution

| Tag | Count |
|-----|-------|
| `<sigh>` | 280 |
| `<laugh>` | 34 |
| `<chuckle>` | 12 |
| `<gasp>` | 3 |

### Voice Characteristics (from analysis)
- Mean F0: 235.5 Hz
- Pitch variation coefficient: 0.351 (highly expressive)
- Dominant prosody: 69.5% emphatic, 30.1% excited

## Training Configuration

| Parameter | v1 | v2 | v3 (current) |
|-----------|-----|------|------|
| Base model | `canopylabs/orpheus-3b-0.1-ft` | same | same |
| Method | LoRA (via Unsloth) | same | same |
| LoRA rank (r) | 64 | 64 | **128** |
| LoRA alpha | 64 | 128 | **256** |
| Target modules | attn + MLP proj | same | **+ embed_tokens, lm_head** |
| Learning rate | 2e-4 (linear) | 5e-5 (cosine) | 5e-5 (cosine) |
| Warmup | 5 steps | 10% of total steps | 10% |
| Batch size | 1 | 1 | 1 |
| Gradient accumulation | 4 steps | 8 steps | 8 steps |
| Epochs | 1 | 5 | 5 |
| Validation eval | none | every 50 steps | every 50 steps |
| Best checkpoint | no | yes (by val loss) | yes |
| Max sequence length | 2048 tokens | 2048 | 2048 |
| Quantization | 4-bit (QLoRA) | same | same |
| Export formats | LoRA only | LoRA + attempt merged | **LoRA + F16 merged + Q8_0 GGUF** |

### v3 rationale
- **LoRA r=128, alpha=256** — doubled capacity for stronger voice adaptation
- **embed_tokens + lm_head** in LoRA targets — adapts the model's input/output embeddings, critical for SNAC audio token representation
- **Prosody-annotated transcripts** — Orpheus tags in training text teach the model to generate appropriate non-speech events
- **Dual export** — both F16 merged model and Q8_0 GGUF for flexible deployment

## Results

### v1 (baseline)

| Metric | Value |
|--------|-------|
| Total training steps | 133 |
| Final training loss | 9.83 |
| Training time | ~3.5 minutes |
| Hardware | NVIDIA RTX A6000 (RunPod) |

### v2/v3 (pending)

Training has not been run yet with v3 data+config. Expected ~287 steps, ~15 minutes on A6000.

## RunPod Quick Start

**Recommended pod:** Any GPU with ≥24GB VRAM (RTX A6000, A100, etc.)

```bash
# Clone the repo (includes training data + pre-trained LoRA via Git LFS)
git clone https://github.com/evity-squibbon/orpheus-lefteye.git /workspace/orpheus-lefteye
cd /workspace/orpheus-lefteye

# Copy data to where train.py expects it
cp -r data/ /workspace/clips/
cp -r metadata/ /workspace/metadata/

# Install dependencies
pip install unsloth datasets soundfile librosa snac accelerate

# Run training (~18 min total: ~11s SNAC encoding + ~17 min training)
python train.py
```

> **Note:** If using a lightweight Docker image (e.g., `nvidia/cuda:12.4.1-devel-ubuntu22.04`), install PyTorch first: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

The training script will:
1. Load the Orpheus 3B model in 4-bit quantization
2. Apply LoRA adapters (r=128, alpha=256, including embed_tokens + lm_head)
3. Load SNAC codec and encode all audio clips to discrete tokens (~10s)
4. Train for 5 epochs with cosine LR schedule
5. Evaluate on validation set every 50 steps
6. Save the best LoRA adapter (by val loss) to `output/lora_adapter/`
7. Export F16 merged model and Q8_0 GGUF

### Using the Pre-trained LoRA

If you just want to use the already-trained adapter (skip training):

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="canopylabs/orpheus-3b-0.1-ft",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load the fine-tuned LoRA
from peft import PeftModel
model = PeftModel.from_pretrained(model, "output/lora_adapter")
```

## Inference (WIP)

Full inference pipeline (text → SNAC tokens → audio) is not yet integrated into this repo. The general flow:

1. Load base model + LoRA adapter
2. Tokenize text prompt with speaker tag: `"lefteye: Your text here"`
3. Generate SNAC token sequence autoregressively
4. Decode SNAC tokens back to audio waveform using the SNAC decoder
5. Save as WAV

Prosody tags can be included in the prompt text to guide expression:
```
"lefteye: Oh my god <gasp> are you serious? <laugh>"
```

See the [Orpheus TTS repo](https://github.com/canopylabs/orpheus-tts) for reference inference code.

## File Structure

```
orpheus-lefteye/
├── README.md                    # This file
├── RESEARCH.md                  # Research notes on Orpheus architecture
├── PROSODY.md                   # v3 prosody annotation methodology
├── train.py                     # Main training script (v3, RunPod-ready)
├── create_hf_dataset.py         # HuggingFace dataset creation utility
├── data/                        # 511 curated WAV clips (Git LFS)
│   ├── batch2_blockparty_making_part1_000.wav
│   ├── ...
│   └── confirmed_chunk_XX_YYY.wav
├── metadata/
│   ├── train.csv                # Training split (459 clips, prosody-annotated)
│   ├── val.csv                  # Validation split (52 clips, prosody-annotated)
│   └── full_metadata.csv        # Full metadata with quality scores + original text
└── output/
    └── lora_adapter/            # Pre-trained LoRA weights (Git LFS)
        ├── adapter_model.safetensors
        ├── adapter_config.json
        └── tokenizer files...
```

## Git LFS

Large files are tracked via Git LFS:
- `*.wav` — all audio clips
- `*.safetensors` — model weights

Make sure you have Git LFS installed (`git lfs install`) before cloning.

## License

This repository is **private** and contains copyrighted audio material used for personal research purposes only. Not for redistribution.
