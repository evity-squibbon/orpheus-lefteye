# Orpheus TTS — Voice Clone

Fine-tuning [Orpheus 3B TTS](https://github.com/canopylabs/orpheus-tts) on a speaking voice using LoRA adapters.

## Overview

This project creates a custom TTS voice by fine-tuning the Orpheus 3B text-to-speech model on audio clips of a speaking voice extracted from interviews, behind-the-scenes footage, and commentary. The result is a LoRA adapter that can generate speech in her voice from arbitrary text.

## Training Data

| Metric | Value |
|--------|-------|
| Total clips | 591 |
| Total duration | 93.5 minutes |
| Train split | 531 clips |
| Validation split | 60 clips |
| Format | 24kHz mono WAV |
| Source | Interviews, commentary, behind-the-scenes footage |

Audio was segmented into 5–20 second clips, transcribed, and aligned. Metadata is in LJSpeech-compatible format.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `canopylabs/orpheus-3b-0.1-ft` |
| Method | LoRA (via Unsloth) |
| LoRA rank (r) | 64 |
| LoRA alpha | 64 |
| Target modules | q, k, v, o, gate, up, down proj |
| Learning rate | 2e-4 (linear schedule) |
| Batch size | 1 |
| Gradient accumulation | 4 steps |
| Epochs | 1 |
| Max sequence length | 2048 tokens |
| Quantization | 4-bit (QLoRA) |
| Optimizer | AdamW 8-bit |

## Results

| Metric | Value |
|--------|-------|
| Total training steps | 133 |
| Final training loss | 9.83 |
| Training time | ~3.5 minutes |
| Hardware | NVIDIA RTX A6000 (RunPod) |

The model converges quickly due to the strong Orpheus base model. Audio tokenization (SNAC encoding all 531 clips) takes ~2 minutes; actual LoRA training takes ~3.5 minutes.

## RunPod Quick Start

**Recommended pod:** Any GPU with ≥24GB VRAM (RTX A6000, A100, etc.)

```bash
# Clone the repo (includes training data + pre-trained LoRA via Git LFS)
git clone https://github.com/evity-squibbon/orpheus-lefteye.git /workspace/orpheus-lefteye
cd /workspace/orpheus-lefteye

# Install dependencies
pip install unsloth datasets soundfile librosa snac accelerate

# Run training (~6 min total: 2 min SNAC encoding + 3.5 min training)
python train.py
```

The training script will:
1. Load the Orpheus 3B model in 4-bit quantization
2. Apply LoRA adapters
3. Load SNAC codec and encode all audio clips to discrete tokens
4. Train for 1 epoch (133 steps)
5. Save the LoRA adapter to `output/lora_adapter/`

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

See the [Orpheus TTS repo](https://github.com/canopylabs/orpheus-tts) for reference inference code.

## File Structure

```
orpheus-lefteye/
├── README.md                    # This file
├── RESEARCH.md                  # Research notes on Orpheus architecture
├── train.py                     # Main training script (RunPod-ready)
├── create_hf_dataset.py         # HuggingFace dataset creation utility
├── data/                        # 591 WAV clips (Git LFS)
│   ├── batch2_blockparty_making_part1_000.wav
│   ├── ...
│   └── confirmed_chunk_XX_YYY.wav
├── metadata/
│   ├── train.csv                # Training split (531 clips)
│   ├── val.csv                  # Validation split (60 clips)
│   ├── metadata.csv             # Full metadata
│   └── metadata_ljspeech.csv    # LJSpeech-format metadata
└── output/
    └── lora_adapter/            # Pre-trained LoRA weights (Git LFS)
        ├── adapter_model.safetensors  (371MB)
        ├── adapter_config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        └── chat_template.jinja
```

## Git LFS

Large files are tracked via Git LFS:
- `*.wav` — all audio clips (~258MB total)
- `*.safetensors` — model weights (~371MB)

Make sure you have Git LFS installed (`git lfs install`) before cloning.

## License

This repository is **private** and contains copyrighted audio material used for personal research purposes only. Not for redistribution.
