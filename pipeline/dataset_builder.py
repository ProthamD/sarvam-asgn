"""
pipeline/dataset_builder.py — Build and publish HuggingFace dataset.

Creates a Dataset with:
  - audio (Audio feature, 16kHz)
  - transcription, language, emotion, style, duration, SNR, etc.
  - 90/10 train/validation split
  - Auto-generated dataset card (README.md)

Pushes to HuggingFace Hub as a public dataset.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from datasets import Dataset, Audio, Value, Features, DatasetDict
from huggingface_hub import HfApi, whoami
from rich.console import Console
from rich.table import Table

from pipeline.utils import (
    get_logger, load_config, get_path, get_env,
    save_json, load_json, seconds_to_hms
)

logger = get_logger("dataset_builder")
console = Console()

# ── HuggingFace schema ────────────────────────────────────────────────────────
FEATURES = Features({
    "segment_id":       Value("string"),
    "audio":            Audio(sampling_rate=16000),
    "transcription":    Value("string"),
    "language":         Value("string"),
    "primary_emotion":  Value("string"),
    "speaking_style":   Value("string"),
    "speech_rate":      Value("string"),
    "duration_seconds": Value("float32"),
    "snr_db":           Value("float32"),
    "source_url":       Value("string"),
    "source_title":     Value("string"),
    "speaker_id":       Value("string"),
    "tag_confidence":   Value("float32"),
    "split":            Value("string"),
})


# ── Dataset card template ─────────────────────────────────────────────────────
DATASET_CARD_TEMPLATE = """---
language:
- en
- hi
tags:
- text-to-speech
- audio
- indian-languages
- speech
pretty_name: Indian TTS 60 Minutes Dataset
size_categories:
- 1K<n<10K
task_categories:
- text-to-speech
license: cc-by-4.0
---

# 🎤 Indian TTS Dataset — 60 Minutes

A curated, high-quality Text-to-Speech (TTS) training dataset containing approximately
**60 minutes** of single-speaker audio from Indian speakers, covering both
**Indian English (en-IN)** and **Hindi (hi-IN)**.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total duration | ~{total_duration} |
| Total segments | {total_segments} |
| Indian English (en-IN) | {en_segments} segments (~{en_duration}) |
| Hindi (hi-IN) | {hi_segments} segments (~{hi_duration}) |
| Average segment length | {avg_duration:.1f}s |
| Average SNR | {avg_snr:.1f} dB |

## Features

| Column | Type | Description |
|--------|------|-------------|
| `segment_id` | string | Unique segment identifier |
| `audio` | Audio (16kHz) | Single-speaker audio clip |
| `transcription` | string | Accurate text transcription |
| `language` | string | Language code (en-IN or hi-IN) |
| `primary_emotion` | string | Detected emotion/tone |
| `speaking_style` | string | Speaking style tag |
| `speech_rate` | string | Pace (slow/normal/fast) |
| `duration_seconds` | float | Clip duration in seconds |
| `snr_db` | float | Signal-to-noise ratio (dB) |
| `source_url` | string | Source YouTube URL |
| `source_title` | string | Source video title |
| `speaker_id` | string | Diarization speaker ID |
| `tag_confidence` | float | LLM tagging confidence |

## Emotion Distribution

{emotion_dist}

## Style Distribution

{style_dist}

## Pipeline

Built with a fully automated pipeline using:
- **yt-dlp** — Audio download from YouTube
- **Sarvam AI Batch ASR** (saaras:v3) — Speaker diarization
- **Sarvam AI REST ASR** (saaras:v3) — Transcription
- **Sarvam LLM** (sarvam-m) — Emotion/style tagging
- Multi-stage quality filtering (SNR, clipping, spectral flatness)

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Attribution required.
Audio sourced from publicly available YouTube content.
"""


def _build_dataset_card(segments: list[dict]) -> str:
    total_dur = sum(s.get("duration_s", 0) for s in segments)
    en_segs = [s for s in segments if "en" in s.get("language_code", "")]
    hi_segs = [s for s in segments if "hi" in s.get("language_code", "")]
    en_dur = sum(s.get("duration_s", 0) for s in en_segs)
    hi_dur = sum(s.get("duration_s", 0) for s in hi_segs)
    avg_dur = total_dur / max(len(segments), 1)
    avg_snr = sum(
        s.get("quality_metrics", {}).get("snr_db", 0)
        for s in segments
    ) / max(len(segments), 1)

    # Emotion distribution
    emotion_counts: dict[str, int] = {}
    for s in segments:
        e = s.get("primary_emotion", "neutral")
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    emotion_dist = "\n".join(
        f"- **{e}**: {c} segments"
        for e, c in sorted(emotion_counts.items(), key=lambda x: -x[1])
    )

    style_counts: dict[str, int] = {}
    for s in segments:
        st = s.get("speaking_style", "conversational")
        style_counts[st] = style_counts.get(st, 0) + 1
    style_dist = "\n".join(
        f"- **{st}**: {c} segments"
        for st, c in sorted(style_counts.items(), key=lambda x: -x[1])
    )

    return DATASET_CARD_TEMPLATE.format(
        total_duration=seconds_to_hms(total_dur),
        total_segments=len(segments),
        en_segments=len(en_segs),
        en_duration=seconds_to_hms(en_dur),
        hi_segments=len(hi_segs),
        hi_duration=seconds_to_hms(hi_dur),
        avg_duration=avg_dur,
        avg_snr=avg_snr,
        emotion_dist=emotion_dist or "- neutral: all segments",
        style_dist=style_dist or "- conversational: all segments",
    )


# ── Video metadata cache ──────────────────────────────────────────────────────

def _load_video_meta_cache(raw_dir: Path) -> dict[str, dict]:
    """Load video metadata for URL/title lookup."""
    manifest_path = raw_dir / "download_manifest.json"
    if manifest_path.exists():
        return load_json(manifest_path)
    return {}


# ── Build and push ────────────────────────────────────────────────────────────

def build_and_push(
    segments: list[dict],
    config: dict | None = None,
    dry_run: bool = False,
) -> str:
    """
    Build HuggingFace Dataset from segments and push to Hub.

    Returns the dataset repo URL.
    """
    cfg = config or load_config()
    raw_dir = get_path("raw_audio", cfg)
    hf_cache_dir = get_path("hf_cache", cfg)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    hf_token = get_env("HF_TOKEN")
    api = HfApi(token=hf_token)
    user_info = whoami(token=hf_token)
    username = user_info["name"]

    dataset_name = cfg["dataset"].get("dataset_name", "indian-tts-60min")
    repo_id = cfg["dataset"].get("hf_repo_id") or f"{username}/{dataset_name}"

    logger.info(f"[build] Building dataset: {repo_id}")
    logger.info(f"  Segments: {len(segments)}")

    # Load video metadata for URL/title
    vid_meta = _load_video_meta_cache(raw_dir)

    # Filter out segments without transcripts or audio
    valid_segments = []
    for seg in segments:
        wav = Path(seg.get("filtered_wav_path") or seg.get("wav_path", ""))
        if not wav.exists():
            logger.warning(f"  Missing audio: {seg['segment_id']} — skipping")
            continue
        if not seg.get("transcript", "").strip():
            logger.warning(f"  Empty transcript: {seg['segment_id']} — skipping")
            continue
        valid_segments.append(seg)

    logger.info(f"  Valid segments (audio + transcript): {len(valid_segments)}")

    total_dur = sum(s.get("duration_s", 0) for s in valid_segments)
    if total_dur < cfg["dataset"]["min_acceptable_duration_min"] * 60:
        logger.warning(
            f"⚠️  Total duration {seconds_to_hms(total_dur)} is below minimum. "
            f"Dataset will still be created."
        )

    # ── Build rows ──────────────────────────────────────────────────────────
    rows = []
    for seg in valid_segments:
        vid_id = seg.get("video_id", "")
        vm = vid_meta.get(vid_id, {})
        wav_path = str(Path(seg.get("filtered_wav_path") or seg["wav_path"]).resolve())

        rows.append({
            "segment_id":       seg["segment_id"],
            "audio":            wav_path,
            "transcription":    seg.get("transcript", ""),
            "language":         seg.get("language_code", ""),
            "primary_emotion":  seg.get("primary_emotion", "neutral"),
            "speaking_style":   seg.get("speaking_style", "conversational"),
            "speech_rate":      seg.get("speech_rate", "normal"),
            "duration_seconds": float(seg.get("duration_s", 0)),
            "snr_db":           float(seg.get("quality_metrics", {}).get("snr_db", 0)),
            "source_url":       vm.get("url", ""),
            "source_title":     vm.get("title", ""),
            "speaker_id":       seg.get("speaker", "SPEAKER_0"),
            "tag_confidence":   float(seg.get("tag_confidence", 0.5)),
            "split":            "",  # filled after split
        })

    # ── Train/validation split ──────────────────────────────────────────────
    split_ratio = cfg["dataset"].get("train_split_ratio", 0.9)
    n_train = int(len(rows) * split_ratio)
    for i, row in enumerate(rows):
        row["split"] = "train" if i < n_train else "validation"

    # Separate into splits
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]

    def rows_to_dict(rows_list):
        if not rows_list:
            return {k: [] for k in rows[0].keys()}
        return {k: [r[k] for r in rows_list] for k in rows_list[0].keys()}

    train_ds = Dataset.from_dict(rows_to_dict(train_rows), features=FEATURES)
    val_ds = Dataset.from_dict(rows_to_dict(val_rows), features=FEATURES)
    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    # ── Print summary table ─────────────────────────────────────────────────
    table = Table(title=f"Dataset: {repo_id}", show_header=True)
    table.add_column("Split", style="cyan")
    table.add_column("Segments", style="green")
    table.add_column("Duration", style="yellow")
    for split_name, split_rows in [("train", train_rows), ("validation", val_rows)]:
        dur = seconds_to_hms(sum(r["duration_seconds"] for r in split_rows))
        table.add_row(split_name, str(len(split_rows)), dur)
    table.add_row(
        "TOTAL", str(len(rows)),
        seconds_to_hms(sum(r["duration_seconds"] for r in rows)),
        style="bold"
    )
    console.print(table)

    if dry_run:
        logger.info("[dry_run] Skipping HuggingFace push.")
        save_json(rows, get_path("metadata", cfg) / "dataset_rows.json")
        return f"(dry_run) {repo_id}"

    # ── Push to Hub ─────────────────────────────────────────────────────────
    logger.info(f"[push] Pushing to HuggingFace Hub: {repo_id}")

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    # Push dataset
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        token=hf_token,
        commit_message="Add TTS dataset via pipeline",
    )

    # Push dataset card
    card = _build_dataset_card(valid_segments)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
        commit_message="Add dataset card",
    )

    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"✅ Dataset published: {repo_url}")
    return repo_url
