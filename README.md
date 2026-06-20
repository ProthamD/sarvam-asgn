# 🎤 TTS Dataset Pipeline

A production-quality pipeline to build a 60-minute Indian TTS training dataset from YouTube videos, published to HuggingFace Hub.

**Coverage:** ~30 min Indian English + ~30 min Hindi  
**APIs:** Sarvam AI (ASR + diarization + LLM) · HuggingFace Datasets

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10+
# FFmpeg must be installed: https://ffmpeg.org/download.html
ffmpeg -version  # verify

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure credentials

Your `.env` file already contains the API keys. Do **not** commit it to git.

```
SARVAM_API_KEY=sk_...
HF_TOKEN=hf_...
```

### 3. Run the full pipeline

```bash
python run_pipeline.py --stage all
```

This runs all 7 stages sequentially. Each stage saves its output, so you can re-run individual stages:

```bash
python run_pipeline.py --stage download     # Stage 1: Download YouTube audio
python run_pipeline.py --stage diarize      # Stage 2: Speaker diarization
python run_pipeline.py --stage segment      # Stage 3: Smart segmentation
python run_pipeline.py --stage filter       # Stage 4: Quality filtering
python run_pipeline.py --stage transcribe   # Stage 5: ASR transcription
python run_pipeline.py --stage tag          # Stage 6: Emotion/style tagging
python run_pipeline.py --stage build        # Stage 7: Build + push HF dataset
```

### 4. Review your data (recommended!)

```bash
# Review all segments interactively
python review_tool.py

# Review a random 20% sample
python review_tool.py --sample 0.2
```

---

## Pipeline Architecture

```
YouTube URLs
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: Download (yt-dlp)                         │
│  → 16kHz mono WAV, video metadata                   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: Diarize (Sarvam Batch API, saaras:v3)     │
│  → Speaker-labeled timestamps                       │
│  → Dominant speaker identification                  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3: Segment (pydub)                           │
│  → Merge adjacent same-speaker segments             │
│  → Split at silence if too long (>90s)              │
│  → Normalize loudness (EBU R128, -23 LUFS)          │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4: Quality Filter (librosa)                  │
│  Gate 1: Duration (15–90s)                          │
│  Gate 2: SNR ≥ 18 dB                               │
│  Gate 3: Clipping < 0.2%                            │
│  Gate 4: Spectral flatness < 0.18 (no music)        │
│  Gate 5: Silence ratio < 35%                        │
│  Gate 6: Speech rate 1.5–18 chars/sec               │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5: Transcribe (Sarvam ASR)                   │
│  ≤25s segments → REST API (synchronous)             │
│  >25s segments → Batch API (async, grouped)         │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 6: Emotion Tag (Sarvam LLM, sarvam-m)        │
│  → primary_emotion, speaking_style, speech_rate     │
│  → Batched 8 transcripts per LLM call               │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 7: Build & Push (HuggingFace Datasets)       │
│  → 90/10 train/validation split                     │
│  → Auto-generated dataset card                      │
│  → Public HuggingFace dataset                       │
└─────────────────────────────────────────────────────┘
```

---

## Configuration

All thresholds are in `config.yaml`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `segmentation.min_duration_s` | 15s | Minimum segment length |
| `segmentation.max_duration_s` | 90s | Maximum segment length |
| `quality.min_snr_db` | 18 dB | Minimum signal-to-noise ratio |
| `quality.max_spectral_flatness` | 0.18 | Music/noise rejection |
| `sarvam.llm_model` | sarvam-m | LLM for emotion tagging |

---

## Data Directory Structure

```
data/
├── raw/                      # Downloaded full-length WAV files
│   ├── <video_id>.wav
│   ├── <video_id>_meta.json
│   └── download_manifest.json
├── segments/                 # Extracted speaker segments
│   └── <video_id>_seg0001.wav
├── filtered/                 # Quality-filtered segments
│   └── <video_id>_seg0001.wav
├── metadata/                 # JSON/JSONL at each stage
│   ├── <video_id>_diarization.json
│   ├── <video_id>_segments.json
│   ├── quality_report.jsonl
│   ├── passed_segments.json
│   ├── transcriptions.jsonl
│   ├── transcribed_segments.json
│   ├── emotion_tags.jsonl
│   ├── tagged_segments.json
│   └── reviewed.jsonl        # Human review decisions
└── hf_upload/                # HuggingFace upload staging
```

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `audio` | Audio (16kHz) | Single-speaker WAV |
| `transcription` | string | Verified text transcription |
| `language` | string | `en-IN` or `hi-IN` |
| `primary_emotion` | string | `neutral\|happy\|sad\|excited\|angry\|calm\|...` |
| `speaking_style` | string | `formal\|informal\|narrative\|...` |
| `speech_rate` | string | `slow\|normal\|fast` |
| `duration_seconds` | float | Clip length |
| `snr_db` | float | Signal-to-noise ratio |
| `source_url` | string | YouTube source URL |
| `source_title` | string | Video title |

---

## Troubleshooting

**FFmpeg not found:**
```bash
# Windows: Install via winget
winget install ffmpeg
# Or download from https://ffmpeg.org/download.html and add to PATH
```

**Sarvam batch job times out:**
- Increase `sarvam.batch_timeout_s` in `config.yaml`
- Or split videos into smaller chunks first

**Not enough segments pass quality filter:**
- Lower `quality.min_snr_db` from 18 to 15
- Increase `quality.max_spectral_flatness` from 0.18 to 0.25
- Add more source YouTube videos: `python run_pipeline.py --add-urls "URL" --lang en-IN`

---

## License

Code: MIT  
Dataset: CC BY 4.0
