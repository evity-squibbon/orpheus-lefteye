# Orpheus TTS Fine-Tuning Research — Left Eye Voice Clone

## Date: 2026-02-18
## Researcher: Clawdbot (subagent)

---

## 1. Orpheus TTS Architecture Overview

Orpheus TTS is built on a **Llama-3B backbone** trained on 100k+ hours of English speech data. It generates **discrete audio tokens** (not raw waveforms) using the **SNAC (Multi-Scale Neural Audio Codec)** at 24kHz. The model is essentially a causal language model that predicts audio token sequences given text input.

### Token Structure
- SNAC encodes audio into **7 tokens per frame** across 3 hierarchical layers:
  - Layer 0: 1 token per frame (coarse)
  - Layer 1: 2 tokens per frame (medium)
  - Layer 2: 4 tokens per frame (fine)
- Token IDs are offset from the text vocabulary: `base_offset = 128266`, with each layer offset by multiples of 4096
- Special tokens: `start_of_human`, `end_of_human`, `start_of_ai`, `end_of_ai`, `start_of_speech`, `end_of_speech`, `pad_token`

### Prompt Format
- **Single speaker (pretrained):** `{text}` → audio tokens
- **Multi-speaker (finetuned-prod):** `{speaker_name}: {text}` → audio tokens
- **Emotion tags:** `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`, `<breathe/>`

---

## 2. Fine-Tuning Data Requirements

### Audio Format
| Parameter | Requirement |
|-----------|-------------|
| **Sample Rate** | **24,000 Hz (24kHz)** — mandatory, SNAC codec operates at this rate |
| **Channels** | **Mono** |
| **Format** | WAV preferred for quality; MP3 acceptable (will be decoded to waveform) |
| **Bit Depth** | 16-bit PCM (standard) |

### Clip Length
| Parameter | Recommendation |
|-----------|----------------|
| **Minimum** | ~3 seconds (shorter clips have too few tokens to learn from) |
| **Sweet Spot** | **5-15 seconds** per clip (based on sample dataset: 3.18s to 13.1s) |
| **Maximum** | 30 seconds recommended; longer clips use more VRAM per sample |
| **Text Length** | ~64 characters / 150-160 text tokens per example |

The official sample dataset (`canopylabs/zac-sample-dataset`) contains clips ranging from **3.18s to 13.1s** with text from 32-140 characters. This is the gold standard for clip format.

### Dataset Size
| Quantity | Expected Quality |
|----------|-----------------|
| **50 examples** | "High quality results" start appearing (official docs) |
| **300 examples/speaker** | "Best results" (official recommendation) |
| **~1,200 samples (3 hours)** | Elise dataset reference — very high quality single-speaker |
| **~80 minutes** | Our dataset — should be excellent with proper segmentation |

With 93.7 minutes of clean audio, we should be able to produce **500-1000+ clips** at 5-12s each — well above the recommended 300.

### Transcription
- **Whisper transcription is sufficient** — the Elise reference dataset uses Whisper transcriptions
- Transcripts should be **normalized**: remove unusual characters, keep punctuation natural
- **Emotion tags** (e.g., `<laugh>`) can be added where present in audio — improves expressiveness
- Text should be plain English with natural formatting

---

## 3. Dataset Structure

### Hugging Face Dataset Format (Required for Orpheus)
The dataset must be a HuggingFace `datasets` object with:
```python
{
    "audio": {  # Audio column (cast to Audio(sampling_rate=24000))
        "array": np.array([...]),  # float32 waveform
        "sampling_rate": 24000
    },
    "text": "The transcript of this audio clip."  # Plain text
}
```

### Preprocessing Pipeline
1. Load audio + text pairs into HF dataset
2. Resample all audio to 24kHz
3. Encode audio through SNAC → get `codes_list` (7 tokens per frame)
4. Remove duplicate frames (consecutive frames where first token matches)
5. Tokenize text with Llama tokenizer
6. Construct `input_ids`:
   ```
   [start_of_human] + text_tokens + [end_of_text] + [end_of_human] + 
   [start_of_ai] + [start_of_speech] + audio_tokens + [end_of_speech] + [end_of_ai]
   ```
7. Set `labels = input_ids`, `attention_mask = [1] * len(input_ids)`

### Directory Layout (Pre-HF Upload)
```
orpheus_training/
├── clips/                   # Individual WAV clips at 24kHz mono
│   ├── clip_0001.wav
│   ├── clip_0002.wav
│   └── ...
├── transcripts/            # Individual text files (backup)
│   ├── clip_0001.txt
│   └── ...
├── metadata/
│   ├── metadata.csv        # filename,text,duration
│   ├── train.csv           # Training split
│   └── val.csv             # Validation split
├── RESEARCH.md             # This file
├── README.md               # Dataset documentation
└── prepare_dataset.py      # Script to create HF dataset + SNAC tokens
```

---

## 4. Training Configuration

### LoRA vs Full Fine-Tune

| Approach | VRAM Needed | Quality | Speed | Recommendation |
|----------|-------------|---------|-------|----------------|
| **LoRA (r=64)** | ~6-8 GB (16-bit) | Excellent for voice cloning | Fast (~20 min on T4 for 60 steps) | ✅ **Recommended** |
| **QLoRA (4-bit)** | ~4-5 GB | Good but slightly lower | Faster | Good fallback |
| **Full Fine-Tune** | ~24+ GB | Best possible | Slow | Only if VRAM allows |

**For our case (~80 min single speaker with 256GB unified memory):** LoRA r=64 is ideal. We have massive VRAM headroom on the Mac Studios, so we could even do full fine-tune, but LoRA gives 97% of the quality at a fraction of the compute.

### Recommended Hyperparameters
```yaml
# Base model
model_name: "canopylabs/orpheus-tts-0.1-pretrained"  # or -finetune-prod

# LoRA config
r: 64
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_alpha: 64
lora_dropout: 0

# Training
learning_rate: 2e-4          # Unsloth recommendation
batch_size: 1                # Per device
gradient_accumulation: 4     # Effective batch = 4
warmup_steps: 5
max_seq_length: 2048         # For clips up to ~15s
optimizer: "adamw_8bit"
weight_decay: 0.001
lr_scheduler: "linear"
epochs: 1-3                  # Start with 1, evaluate
```

### VRAM Requirements
- **Orpheus 3B + LoRA (16-bit):** ~6-8 GB VRAM on CUDA
- **Apple Silicon (Metal):** Untested officially, but with 256GB unified memory, this should work fine via MLX or PyTorch MPS
- **SNAC encoding:** ~2 GB additional during data preprocessing

### Training Duration Estimates
- **~1000 clips, 1 epoch, batch=1, grad_accum=4:** ~250 steps, ~15-30 min on T4 GPU
- **On Apple Silicon M2 Ultra:** Likely 2-4x slower than CUDA due to Metal backend; ~30-60 min estimated

---

## 5. Chatterbox TTS (Backup Option)

### Overview
Chatterbox is by Resemble AI, available in Standard (Llama-based, 500M), Turbo (GPT-2 based, 350M), and Multilingual (500M) variants.

### Fine-Tuning Requirements

| Parameter | Chatterbox Requirement |
|-----------|----------------------|
| **Audio Format** | WAV, mono |
| **Sample Rate** | 16kHz (resampled automatically) |
| **Output Rate** | 24kHz (vocoder output) |
| **Clip Length** | **3-10 seconds** optimal |
| **Min Dataset** | 30+ minutes for basic training |
| **Recommended** | 1 hour of clean audio |
| **VRAM** | ≥18 GB for LoRA |

### Dataset Format (LJSpeech)
```
MyTTSDataset/
├── metadata.csv        # filename|raw_text|normalized_text  (pipe-delimited)
└── wavs/
    ├── recording_001.wav
    └── ...
```

### Key Differences from Orpheus
1. **No SNAC tokenization needed** — Chatterbox uses its own S3 tokenizer internally
2. **Simpler data prep** — just WAV files + metadata CSV
3. **Lower VRAM** — Turbo model is only 350M params
4. **Zero-shot cloning built-in** — uses `audio_prompt_path` for voice reference
5. **Separate fine-tuning repo:** github.com/gokhaneraslan/chatterbox-finetuning

### Verdict
Chatterbox is simpler to fine-tune but Orpheus has better voice quality and expressiveness. **Primary: Orpheus. Backup: Chatterbox.**

---

## 6. Data Preparation Plan for Left Eye Corpus

### Source Audio Summary
| Batch | Files | Duration | Format | Quality |
|-------|-------|----------|--------|---------|
| confirmed_chunks | 6 WAV | ~675s (11.25 min) | WAV 44.1kHz 16-bit mono | Highest (documentary) |
| pass3_clips | 5 MP3 | ~326s (5.4 min) | MP3 (various rates) | High (isolated vocal) |
| batch2/final | 17 MP3 | ~26.5 min | MP3 128kbps 16kHz mono | Good (mixed sources) |
| batch3/final | 15 MP3 | ~29.5 min | MP3 128kbps | Good (TV interviews) |
| batch4/final | 23 MP3 | ~20.9 min | MP3 128kbps | Good (short clips) |
| **TOTAL** | **66 files** | **~93.7 min** | | |

### Processing Pipeline
1. **Segment long files** into 5-12 second clips at sentence boundaries using VAD (Voice Activity Detection)
2. **Convert all clips** to 24kHz mono WAV 16-bit PCM
3. **Transcribe** each clip with Whisper (large-v3-turbo via mlx_whisper for speed)
4. **Quality filter**: Remove clips shorter than 3s or with poor transcription confidence
5. **Create metadata.csv** with columns: `filename`, `text`, `duration`
6. **Split** into train (90%) and val (10%) sets
7. **Create HuggingFace dataset** format for Orpheus pipeline

### Expected Output
- **~700-1000 clips** at 5-12s average
- All at 24kHz mono WAV
- Whisper transcriptions for each
- Ready for SNAC tokenization during training

---

## 7. Mac-Specific Considerations

### Training on Apple Silicon
- Orpheus training officially uses CUDA (Flash Attention 2, etc.)
- **Unsloth does NOT support Apple Silicon** — it requires CUDA
- **Options for training on Mac:**
  1. Use vanilla HuggingFace Trainer with MPS backend (PyTorch)
  2. Use MLX for LoRA fine-tuning (if Orpheus is ported)
  3. Use cloud GPU (Colab, Lambda, etc.) — **most reliable option**
  4. Run inference locally after training remotely

### SNAC Encoding on Mac
- SNAC model works on CPU (slower) or MPS (faster)
- Data preprocessing can be done locally on Mac
- Encode all audio to tokens locally, then train on cloud GPU

### Recommended Workflow
1. **Prepare data locally** (segment, convert, transcribe) — Mac Studio
2. **SNAC-encode locally** (if needed, or during remote training)
3. **Upload HF dataset** to HuggingFace Hub
4. **Train on cloud GPU** (Colab free T4, or Lambda A100)
5. **Download adapter weights** and run inference locally via llama.cpp

---

## 8. References

- [Orpheus TTS GitHub](https://github.com/canopyai/Orpheus-TTS)
- [Orpheus Sample Dataset](https://huggingface.co/datasets/canopylabs/zac-sample-dataset)
- [Unsloth TTS Fine-tuning Guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning)
- [Unsloth Orpheus Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)
- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [Chatterbox Fine-tuning Kit](https://github.com/gokhaneraslan/chatterbox-finetuning)
- [Elise TTS Dataset](https://huggingface.co/datasets/MrDragonFox/Elise) (reference single-speaker dataset)
