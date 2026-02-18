#!/usr/bin/env python3
"""
Create a HuggingFace dataset from the prepared clips and transcriptions.
This creates the dataset in the format expected by Orpheus TTS fine-tuning.

Usage:
    python3 create_hf_dataset.py                     # Create local dataset
    python3 create_hf_dataset.py --push --repo USER/REPO  # Push to HF Hub
"""

import argparse
import csv
import os
from pathlib import Path

# Try to import datasets; if not available, guide installation
try:
    from datasets import Dataset, Audio, DatasetDict
except ImportError:
    print("Error: 'datasets' package not installed.")
    print("Install with: pip install datasets soundfile librosa")
    exit(1)

CLIPS_DIR = Path("/Volumes/SSD/openclaw/projects/lefteye_voice/orpheus_training/clips")
METADATA_DIR = Path("/Volumes/SSD/openclaw/projects/lefteye_voice/orpheus_training/metadata")
OUTPUT_DIR = Path("/Volumes/SSD/openclaw/projects/lefteye_voice/orpheus_training/hf_dataset")

def load_metadata(csv_path):
    """Load metadata from CSV."""
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = CLIPS_DIR / row["filename"]
            if wav_path.exists():
                entries.append({
                    "audio": str(wav_path),
                    "text": row["text"],
                    "duration": float(row["duration"])
                })
    return entries

def create_dataset(split_name, csv_path):
    """Create a HuggingFace dataset from a metadata CSV."""
    entries = load_metadata(csv_path)
    if not entries:
        print(f"  Warning: No entries found in {csv_path}")
        return None
    
    dataset = Dataset.from_dict({
        "audio": [e["audio"] for e in entries],
        "text": [e["text"] for e in entries],
        "duration": [e["duration"] for e in entries],
    })
    
    # Cast audio column to Audio feature with 24kHz sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
    
    print(f"  {split_name}: {len(dataset)} samples")
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Create HuggingFace dataset for Orpheus TTS")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--repo", type=str, help="HF repo name (e.g., username/lefteye-voice)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Local output directory")
    args = parser.parse_args()
    
    print("Creating HuggingFace dataset for Orpheus TTS fine-tuning...")
    print(f"  Clips directory: {CLIPS_DIR}")
    print(f"  Metadata directory: {METADATA_DIR}")
    
    # Create train and val datasets
    train_ds = create_dataset("train", METADATA_DIR / "train.csv")
    val_ds = create_dataset("val", METADATA_DIR / "val.csv")
    
    if train_ds is None:
        print("Error: Could not create training dataset")
        return
    
    # Create DatasetDict
    dataset_dict = DatasetDict({"train": train_ds})
    if val_ds is not None:
        dataset_dict["validation"] = val_ds
    
    # Save locally
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    print(f"\n  Saved to: {output_path}")
    
    # Optionally push to Hub
    if args.push:
        if not args.repo:
            print("Error: --repo required when using --push")
            return
        print(f"\n  Pushing to HuggingFace Hub: {args.repo}")
        dataset_dict.push_to_hub(args.repo, private=True)
        print("  Done!")
    
    # Print dataset info
    print(f"\nDataset summary:")
    print(f"  Train samples: {len(train_ds)}")
    if val_ds:
        print(f"  Val samples: {len(val_ds)}")
    print(f"  Features: {train_ds.features}")
    
    # Show a sample
    print(f"\nSample entry:")
    sample = train_ds[0]
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Audio SR: {sample['audio']['sampling_rate']}")
    print(f"  Audio shape: {len(sample['audio']['array'])} samples "
          f"({len(sample['audio']['array'])/24000:.1f}s)")
    print(f"  Duration: {sample['duration']}s")

if __name__ == "__main__":
    main()
