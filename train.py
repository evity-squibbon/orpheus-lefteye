#!/usr/bin/env python3
"""Orpheus TTS Fine-Tuning Script for Left Eye Voice Clone
   v2 — multi-epoch, cosine schedule, validation eval, best checkpoint saving
"""

import os, csv, torch, numpy as np
from pathlib import Path
from tqdm import tqdm

# ---- Config ----
# Auto-detect: repo clone vs flat /workspace layout
SCRIPT_DIR = Path(__file__).parent.resolve()
if (SCRIPT_DIR / "data").exists():
    # Running from repo root
    CLIPS_DIR = SCRIPT_DIR / "data"
    METADATA_DIR = SCRIPT_DIR / "metadata"
else:
    # Legacy flat layout
    CLIPS_DIR = Path("/workspace/clips")
    METADATA_DIR = Path("/workspace/metadata")

OUTPUT_DIR = Path("/workspace/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Training Hyperparameters ----
MAX_SEQ_LENGTH = 2048
SPEAKER = "lefteye"
NUM_EPOCHS = 5             # v1 was 1 — more passes = better voice capture
LEARNING_RATE = 5e-5       # v1 was 2e-4 — lower LR + more epochs = smoother convergence
LR_SCHEDULER = "cosine"    # v1 was linear — cosine decays more gracefully
WARMUP_RATIO = 0.1         # 10% warmup instead of fixed 5 steps
GRAD_ACCUM = 8             # v1 was 4 — larger effective batch (8) for stability
LORA_R = 64
LORA_ALPHA = 128           # v1 was 64 — 2x rank for stronger adaptation signal
EVAL_STEPS = 50            # evaluate on val set every 50 steps
SAVE_TOTAL_LIMIT = 5
LOGGING_STEPS = 10

print("=" * 60)
print("Orpheus TTS Fine-Tuning — Left Eye Voice Clone (v2)")
print("=" * 60)
print(f"  Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}, Schedule: {LR_SCHEDULER}")
print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, Grad accum: {GRAD_ACCUM}")
print(f"  Warmup: {WARMUP_RATIO*100:.0f}%, Eval every {EVAL_STEPS} steps")

# ---- Step 1: Load model with Unsloth ----
print("\n[1/7] Loading Orpheus model with Unsloth...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="canopylabs/orpheus-3b-0.1-ft",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # auto
    load_in_4bit=True,
)

# ---- Step 2: Apply LoRA ----
print("\n[2/7] Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ---- Step 3: Load SNAC codec ----
print("\n[3/7] Loading SNAC codec for audio tokenization...")
from snac import SNAC
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

# ---- Step 4: Prepare dataset ----
print("\n[4/7] Preparing dataset (SNAC encoding audio)...")
import soundfile as sf

def encode_audio_to_snac_tokens(audio_path):
    """Convert a WAV file to SNAC discrete tokens."""
    audio, sr = sf.read(audio_path)
    if sr != 24000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    
    with torch.no_grad():
        codes = snac_model.encode(audio_tensor)
    
    c0 = codes[0][0].cpu().tolist()
    c1 = codes[1][0].cpu().tolist()
    c2 = codes[2][0].cpu().tolist()
    
    flat_tokens = []
    for i in range(len(c0)):
        flat_tokens.append(c0[i])
        idx1 = i * 2
        if idx1 < len(c1):
            flat_tokens.append(c1[idx1] + 4096)
        if idx1 + 1 < len(c1):
            flat_tokens.append(c1[idx1 + 1] + 4096)
        idx2 = i * 4
        for j in range(4):
            if idx2 + j < len(c2):
                flat_tokens.append(c2[idx2 + j] + 8192)
    
    return flat_tokens

# Special token IDs
start_of_human = tokenizer.convert_tokens_to_ids("<|start_of_human|>") if "<|start_of_human|>" in tokenizer.get_vocab() else None
end_of_human = tokenizer.convert_tokens_to_ids("<|end_of_human|>") if "<|end_of_human|>" in tokenizer.get_vocab() else None
start_of_ai = tokenizer.convert_tokens_to_ids("<|start_of_ai|>") if "<|start_of_ai|>" in tokenizer.get_vocab() else None
end_of_ai = tokenizer.convert_tokens_to_ids("<|end_of_ai|>") if "<|end_of_ai|>" in tokenizer.get_vocab() else None
start_of_speech = tokenizer.convert_tokens_to_ids("<|start_of_speech|>") if "<|start_of_speech|>" in tokenizer.get_vocab() else None
end_of_speech = tokenizer.convert_tokens_to_ids("<|end_of_speech|>") if "<|end_of_speech|>" in tokenizer.get_vocab() else None

print(f"  Special tokens: human={start_of_human}, ai={start_of_ai}, speech={start_of_speech}")

vocab = tokenizer.get_vocab()
special_tokens = {k: v for k, v in vocab.items() if "custom_token" in k.lower() or "start_of" in k.lower() or "end_of" in k.lower()}
print(f"  Found {len(special_tokens)} special/custom tokens")

def process_samples(samples, desc="Processing"):
    """SNAC encode a list of samples, return processed list."""
    processed = []
    skipped = 0
    for sample in tqdm(samples, desc=desc):
        try:
            audio_path = CLIPS_DIR / sample["filename"]
            if not audio_path.exists():
                skipped += 1
                continue
            
            snac_tokens = encode_audio_to_snac_tokens(audio_path)
            text_prompt = f"{SPEAKER}: {sample['text']}"
            text_ids = tokenizer.encode(text_prompt, add_special_tokens=False)
            
            if start_of_human is not None:
                input_ids = [start_of_human] + text_ids + [end_of_human, start_of_ai, start_of_speech] + snac_tokens + [end_of_speech, end_of_ai]
            else:
                input_ids = text_ids + snac_tokens
            
            if len(input_ids) > MAX_SEQ_LENGTH:
                input_ids = input_ids[:MAX_SEQ_LENGTH]
            
            processed.append({
                "input_ids": input_ids,
                "labels": input_ids.copy(),
            })
        except Exception as e:
            print(f"  Error processing {sample.get('filename', '?')}: {e}")
            skipped += 1
    
    print(f"  Processed: {len(processed)}, Skipped: {skipped}")
    return processed

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

processed_train = process_samples(train_samples, "SNAC encoding train")
processed_val = process_samples(val_samples, "SNAC encoding val")

print(f"  Avg train seq length: {np.mean([len(s['input_ids']) for s in processed_train]):.0f} tokens")
if processed_val:
    print(f"  Avg val seq length: {np.mean([len(s['input_ids']) for s in processed_val]):.0f} tokens")

# ---- Step 5: Configure training ----
print("\n[5/7] Configuring trainer...")
from trl import SFTTrainer, SFTConfig
from torch.utils.data import Dataset as TorchDataset

class OrpheusDataset(TorchDataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = OrpheusDataset(processed_train)
val_dataset = OrpheusDataset(processed_val) if processed_val else None

def collate_fn(batch):
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

total_steps = (len(train_dataset) // GRAD_ACCUM) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    lr_scheduler_type=LR_SCHEDULER,
    optim="adamw_8bit",
    logging_steps=LOGGING_STEPS,
    save_steps=EVAL_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field=None,
    report_to="none",
    # Validation
    eval_strategy="steps" if val_dataset else "no",
    eval_steps=EVAL_STEPS if val_dataset else None,
    load_best_model_at_end=True if val_dataset else False,
    metric_for_best_model="eval_loss" if val_dataset else None,
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    tokenizer=tokenizer,
)

# ---- Step 6: Train ----
print(f"\n[6/7] Starting training...")
print(f"  {NUM_EPOCHS} epochs, ~{total_steps} steps, warmup: {warmup_steps}")
print(f"  Effective batch size: {GRAD_ACCUM}")
print(f"  Eval every {EVAL_STEPS} steps on {len(processed_val)} val samples")

trainer.train()

# Print final metrics
if trainer.state.log_history:
    train_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    val_losses = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
    if train_losses:
        print(f"\n  Final train loss: {train_losses[-1]:.4f}")
        print(f"  Min train loss: {min(train_losses):.4f}")
    if val_losses:
        print(f"  Final val loss: {val_losses[-1]:.4f}")
        print(f"  Best val loss: {min(val_losses):.4f}")

# ---- Step 7: Save ----
print("\n[7/7] Saving model...")
model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
print(f"  LoRA adapter saved to {OUTPUT_DIR / 'lora_adapter'}")

try:
    print("  Merging LoRA into base model...")
    model.save_pretrained_merged(str(OUTPUT_DIR / "merged_model"), tokenizer, save_method="merged_16bit")
    print(f"  Merged model saved to {OUTPUT_DIR / 'merged_model'}")
except Exception as e:
    print(f"  Merge failed (non-critical): {e}")

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
