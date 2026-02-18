#!/usr/bin/env python3
"""Orpheus TTS Fine-Tuning Script for Left Eye Voice Clone"""

import os, csv, torch, numpy as np
from pathlib import Path
from tqdm import tqdm

# ---- Config ----
DATA_DIR = Path("/workspace")
CLIPS_DIR = DATA_DIR / "clips"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SEQ_LENGTH = 2048
SPEAKER = "lefteye"

print("=" * 60)
print("Orpheus TTS Fine-Tuning — Left Eye Voice Clone")
print("=" * 60)

# ---- Step 1: Load model with Unsloth ----
print("\n[1/6] Loading Orpheus model with Unsloth...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="canopylabs/orpheus-3b-0.1-ft",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # auto
    load_in_4bit=True,
)

# ---- Step 2: Apply LoRA ----
print("\n[2/6] Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ---- Step 3: Load SNAC codec ----
print("\n[3/6] Loading SNAC codec for audio tokenization...")
from snac import SNAC
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

# ---- Step 4: Prepare dataset ----
print("\n[4/6] Preparing dataset (SNAC encoding audio)...")
import soundfile as sf

def encode_audio_to_snac_tokens(audio_path):
    """Convert a WAV file to SNAC discrete tokens."""
    audio, sr = sf.read(audio_path)
    if sr != 24000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    
    # SNAC expects [batch, 1, samples]
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    
    with torch.no_grad():
        codes = snac_model.encode(audio_tensor)
    
    # codes is a list of 3 tensors (multi-scale): [B, T1], [B, T2], [B, T3]
    # Interleave them into a flat token sequence with offsets
    # Layer 0: tokens as-is (0-4095)
    # Layer 1: tokens + 4096
    # Layer 2: tokens + 8192  
    c0 = codes[0][0].cpu().tolist()  # coarsest
    c1 = codes[1][0].cpu().tolist()
    c2 = codes[2][0].cpu().tolist()
    
    # Interleave: for each coarse token, add 2 medium + 4 fine tokens
    flat_tokens = []
    for i in range(len(c0)):
        flat_tokens.append(c0[i])
        # Medium tokens (2 per coarse)
        idx1 = i * 2
        if idx1 < len(c1):
            flat_tokens.append(c1[idx1] + 4096)
        if idx1 + 1 < len(c1):
            flat_tokens.append(c1[idx1 + 1] + 4096)
        # Fine tokens (4 per coarse) 
        idx2 = i * 4
        for j in range(4):
            if idx2 + j < len(c2):
                flat_tokens.append(c2[idx2 + j] + 8192)
    
    return flat_tokens

# Special token IDs from the Orpheus tokenizer
start_of_human = tokenizer.convert_tokens_to_ids("<|start_of_human|>") if "<|start_of_human|>" in tokenizer.get_vocab() else None
end_of_human = tokenizer.convert_tokens_to_ids("<|end_of_human|>") if "<|end_of_human|>" in tokenizer.get_vocab() else None
start_of_ai = tokenizer.convert_tokens_to_ids("<|start_of_ai|>") if "<|start_of_ai|>" in tokenizer.get_vocab() else None
end_of_ai = tokenizer.convert_tokens_to_ids("<|end_of_ai|>") if "<|end_of_ai|>" in tokenizer.get_vocab() else None
start_of_speech = tokenizer.convert_tokens_to_ids("<|start_of_speech|>") if "<|start_of_speech|>" in tokenizer.get_vocab() else None
end_of_speech = tokenizer.convert_tokens_to_ids("<|end_of_speech|>") if "<|end_of_speech|>" in tokenizer.get_vocab() else None

# Print special token mapping
print(f"  Special tokens: human={start_of_human}, ai={start_of_ai}, speech={start_of_speech}")

# If special tokens not found, use the format from the notebook
# The notebook uses: text prompt -> audio tokens with <custom_token_X> format
vocab = tokenizer.get_vocab()
special_tokens = {k: v for k, v in vocab.items() if "custom_token" in k.lower() or "start_of" in k.lower() or "end_of" in k.lower()}
print(f"  Found {len(special_tokens)} special/custom tokens")
if special_tokens:
    # Print first 10
    for k, v in list(special_tokens.items())[:10]:
        print(f"    {k}: {v}")

# Load metadata
train_samples = []
with open(METADATA_DIR / "train.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        train_samples.append(row)

val_samples = []
with open(METADATA_DIR / "val.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        val_samples.append(row)

print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

# Process training data — encode all audio to SNAC tokens
processed_train = []
skipped = 0
for i, sample in enumerate(tqdm(train_samples, desc="SNAC encoding train")):
    try:
        audio_path = CLIPS_DIR / sample["filename"]
        if not audio_path.exists():
            skipped += 1
            continue
        
        snac_tokens = encode_audio_to_snac_tokens(audio_path)
        
        # Format the prompt
        text_prompt = f"{SPEAKER}: {sample['text']}"
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=False)
        
        # Build full sequence: text + audio tokens
        # Using the Orpheus format from the notebook
        if start_of_human is not None:
            input_ids = [start_of_human] + text_ids + [end_of_human, start_of_ai, start_of_speech] + snac_tokens + [end_of_speech, end_of_ai]
        else:
            # Fallback: just concatenate text + audio
            input_ids = text_ids + snac_tokens
        
        # Truncate if needed
        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
        
        processed_train.append({
            "input_ids": input_ids,
            "labels": input_ids.copy(),
            "attention_mask": [1] * len(input_ids),
        })
        
    except Exception as e:
        print(f"  Error processing {sample['filename']}: {e}")
        skipped += 1

print(f"  Processed: {len(processed_train)}, Skipped: {skipped}")
print(f"  Avg sequence length: {np.mean([len(s['input_ids']) for s in processed_train]):.0f} tokens")

# ---- Step 5: Train ----
print("\n[5/6] Starting training...")
from trl import SFTTrainer, SFTConfig

# Create a simple dataset class
from torch.utils.data import Dataset as TorchDataset

class OrpheusDataset(TorchDataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = OrpheusDataset(processed_train)

# Data collator for padding
def collate_fn(batch):
    # Handle both dict and non-dict items
    if isinstance(batch[0], dict):
        input_ids_list = [b["input_ids"] for b in batch]
        labels_list = [b.get("labels", b["input_ids"]) for b in batch]
    else:
        input_ids_list = [b for b in batch]
        labels_list = [b for b in batch]
    
    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id or 0
    
    input_ids = []
    labels = []
    attention_mask = []
    
    for ids, labs in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        labels.append(labs + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
    
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_steps=5,
    weight_decay=0.001,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field=None,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    tokenizer=tokenizer,
)

print(f"  Training for {training_args.num_train_epochs} epoch(s)...")
print(f"  Steps: ~{len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

trainer.train()

# ---- Step 6: Save ----
print("\n[6/6] Saving model...")
model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
print(f"  LoRA adapter saved to {OUTPUT_DIR / 'lora_adapter'}")

# Try to save merged model
try:
    print("  Merging LoRA into base model...")
    model.save_pretrained_merged(str(OUTPUT_DIR / "merged_model"), tokenizer, save_method="merged_16bit")
    print(f"  Merged model saved to {OUTPUT_DIR / 'merged_model'}")
except Exception as e:
    print(f"  Merge failed (non-critical): {e}")

# Try GGUF export
try:
    print("  Exporting GGUF (q8_0)...")
    model.save_pretrained_gguf(str(OUTPUT_DIR / "gguf"), tokenizer, quantization_method="q8_0")
    print(f"  GGUF saved to {OUTPUT_DIR / 'gguf'}")
except Exception as e:
    print(f"  GGUF export failed (non-critical): {e}")

print("\n" + "=" * 60)
print("Training complete!")
print(f"  LoRA adapter: {OUTPUT_DIR / 'lora_adapter'}")
print(f"  Checkpoints: {OUTPUT_DIR / 'checkpoints'}")
print("=" * 60)
