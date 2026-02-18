# Prosody Annotation Pipeline — v3 Dataset

## Overview

The v3 dataset uses a multi-stage pipeline to detect non-speech events and prosodic features in Left Eye's voice recordings, then annotates transcriptions with Orpheus TTS emotion/prosody tags.

## Methodology

### Phase 1: Re-Transcription (Whisper large-v3)

All 591 source clips were re-transcribed using `mlx-whisper` with OpenAI's `whisper-large-v3` model (via MLX acceleration on Apple Silicon).

**Configuration:**
- Model: `whisper-large-v3` (MLX-optimized)
- Language: English (forced)
- Word-level timestamps: enabled
- Condition on previous text: disabled

**Results:**
- 591 clips transcribed in 8.2 minutes (1.2 clips/sec on GPU)
- 468 transcription differences found vs. original v1 transcripts
- Key improvements: proper punctuation, capitalization, and more accurate word boundaries
- Word-level timestamps enable precise prosody tag placement

### Phase 2: Data Curation

Clips were quality-scored across three dimensions and filtered.

**Quality Metrics:**
- **Clarity (40% weight):** SNR estimation, spectral flatness (lower = more tonal/speech-like), Whisper word probability, no-speech probability
- **Expressiveness (35% weight):** Pitch variation coefficient (F0 std/mean), energy dynamic range, F0 range, speaking rate
- **Length appropriateness (25% weight):** Optimal 3-15s, acceptable 2-20s

**Removal criteria:**
- RMS energy < -45dB (too quiet for reliable learning)
- Quality score < 0.85 combined threshold

**Results:**
- 35 clips removed: too quiet (all from `batch3_lefteye_tupac` series — likely low-quality recording)
- 45 clips removed: quality score < 0.85 threshold
- **511 clips retained** (from 591 original)

### Phase 3: Prosody Annotation

Three analysis methods combined to detect non-speech events and annotate transcripts.

#### 3a: Audio Event Detection (Spectral Analysis)

Non-speech events detected by analyzing gaps between words (using word-level timestamps from Phase 1).

**Detection methods per event type:**

| Event | Detection Criteria | Confidence Threshold |
|-------|-------------------|---------------------|
| **Laughter** | Rhythmic energy oscillation (>30% sign changes), mid-frequency centroid (800-4000 Hz), high ZCR, moderate energy | 0.50 for `<laugh>`, <0.50 for `<chuckle>` |
| **Breathing** | Low energy (below mean), high spectral flatness (>0.3, noise-like), low centroid (<2000 Hz), duration 0.15-1.0s | 0.35 |
| **Sigh** | Moderate energy, falling energy envelope (first half > second half × 1.3), duration 0.4-2.5s | 0.30 |
| **Gasp** | Sudden energy onset (latter 2/3 > first 1/3 × 2), short duration (0.1-0.6s), high centroid (>1500 Hz) | 0.25 |

**Important design decisions:**
- Events are only detected in **gaps between words** (inter-word silence), not during speech
- Minimum confidence of 0.35 required for tag insertion
- No tags are hallucinated — if no audio evidence exists, the transcript is left as-is

#### 3b: Prosodic Feature Extraction

Per-clip features extracted using custom autocorrelation-based pitch estimation (librosa's `pyin` segfaults on this macOS/ARM build):

- **Pitch (F0):** Autocorrelation-based, 60-500 Hz range, confidence threshold 0.3
- **Energy:** RMS envelope, dynamic range, emphasis point detection
- **Speaking rate:** Estimated from word timestamps (syllables/sec)
- **Pitch contour:** Linear regression on F0 track → flat/rising/falling/varied

**Prosody classifications:**
| Class | Criteria |
|-------|---------|
| Excited | PVC > 0.2, EDR > 25, speaking rate > 5 syl/s |
| Emphatic | PVC > 0.15, EDR > 15 |
| Animated | Speaking rate > 5, PVC > 0.1 |
| Calm | PVC < 0.08, speaking rate < 3 |
| Contemplative | PVC < 0.05, EDR < 10, falling pitch |
| Neutral | Default |

#### 3c: Tag Placement

Tags are inserted at the word boundary closest to where the audio event was detected:
- Events before the first word → tag placed at start of transcript
- Events after the last word → tag placed at end of transcript
- Events between words → tag placed after the preceding word

### Supported Orpheus Tags

| Tag | Meaning | Count in Dataset |
|-----|---------|-----------------|
| `<laugh>` | Full laughter | 34 |
| `<chuckle>` | Light/brief laughter | 12 |
| `<sigh>` | Audible sigh | 280 |
| `<gasp>` | Sharp breath intake | 3 |
| `<breathe/>` | Audible breath | 0 (below threshold) |
| `<cough>` | Cough | 0 (not detected) |
| `<sniffle>` | Sniffle | 0 (not detected) |
| `<groan>` | Groan | 0 (not detected) |
| `<yawn>` | Yawn | 0 (not detected) |

**Note:** The dominance of `<sigh>` tags reflects Left Eye's natural speech patterns in interviews — she frequently uses breath pauses with falling energy that match sigh detection criteria. Manual spot-checking confirmed these are reasonable annotations (audible breath releases between phrases).

## Voice Characteristics

Based on prosodic analysis of 511 curated clips:

- **Mean F0:** 235.5 Hz (typical for adult female)
- **F0 Range:** 118.6 - 381.1 Hz
- **Mean Pitch Variation Coefficient:** 0.351 (high — very expressive speaker)
- **Dominant Prosody:** 69.5% emphatic, 30.1% excited
- **Total Duration:** 81.8 minutes

## Tools & Environment

- **Transcription:** mlx-whisper (whisper-large-v3 on Apple Silicon M2 Ultra)
- **Audio Analysis:** librosa 0.10.x (spectral features), custom autocorrelation (pitch)
- **Processing:** Python 3.12, sequential per clip (~0.12s/clip for prosody)
- **Hardware:** Mac Studio M2 Ultra, 256GB unified memory

## Quality Safeguards

1. **No hallucinated tags:** Every tag is backed by audio evidence in a detected inter-word gap
2. **Confidence filtering:** Minimum 0.35 confidence for any tag insertion
3. **Conservative defaults:** When in doubt, transcript is left un-annotated
4. **Spot-checked results:** 10 random clips verified for correct playback and sensible annotations
5. **Comparison logging:** All differences between v1 and v3 transcriptions logged for review
