"""
pipeline/segmenter.py — Smart audio segmentation.

Given diarization output (speaker-labeled timestamps), this module:
  1. Filters to the dominant speaker only.
  2. Merges adjacent dominant-speaker segments within a configurable gap.
  3. Splits merged segments that exceed max_duration at natural silence
     boundaries (using pydub's silence detection).
  4. Trims leading/trailing silence, applies fade-in/out.
  5. Normalizes loudness to EBU R128 (-23 LUFS).
  6. Exports final segments as 16 kHz mono WAV.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_silence
from rich.progress import track

from pipeline.utils import (
    get_logger, load_config, get_path, save_json, load_json,
    append_jsonl, seconds_to_hms
)

logger = get_logger("segmenter")


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _load_audio_segment(wav_path: str, start_ms: int, end_ms: int) -> AudioSegment:
    """Load a slice of a WAV file as a pydub AudioSegment."""
    audio = AudioSegment.from_wav(wav_path)
    return audio[start_ms:end_ms]


def _normalize_loudness(seg: AudioSegment, target_lufs: float = -23.0) -> AudioSegment:
    """Normalize to target dBFS (approximate LUFS normalization)."""
    current_dbfs = seg.dBFS
    if current_dbfs == float("-inf"):
        return seg
    diff = target_lufs - current_dbfs
    # Clamp gain to avoid over-amplification of very quiet segments
    diff = max(-20.0, min(diff, 20.0))
    return seg.apply_gain(diff)


def _apply_fade(seg: AudioSegment, fade_ms: int) -> AudioSegment:
    """Apply fade-in and fade-out to prevent clicks."""
    return seg.fade_in(fade_ms).fade_out(fade_ms)


def _trim_silence(seg: AudioSegment, silence_thresh_dbfs: int = -50) -> AudioSegment:
    """Trim leading and trailing silence."""
    # detect non-silent regions
    non_silent = []
    chunk_size = 10  # ms
    for i in range(0, len(seg), chunk_size):
        chunk = seg[i:i + chunk_size]
        if chunk.dBFS > silence_thresh_dbfs:
            non_silent.append(i)
    if not non_silent:
        return seg
    start = max(0, non_silent[0] - 50)
    end = min(len(seg), non_silent[-1] + chunk_size + 50)
    return seg[start:end]


# ── Segment merging ───────────────────────────────────────────────────────────

def _merge_segments(
    speaker_segments: list[dict],
    dominant_speaker: str,
    max_gap_ms: int,
    min_dur_s: float,
    max_dur_s: float,
) -> list[tuple[float, float]]:
    """
    Filter to dominant speaker, merge nearby segments, return (start_s, end_s) list.
    """
    # Filter to dominant speaker only
    dom_segs = [
        s for s in speaker_segments
        if s["speaker"] == dominant_speaker
        and (s["end"] - s["start"]) >= 1.0  # skip sub-1s fragments
    ]

    if not dom_segs:
        return []

    # Sort by start time
    dom_segs = sorted(dom_segs, key=lambda x: x["start"])

    # Merge adjacent segments within max_gap_ms
    merged: list[tuple[float, float]] = []
    cur_start = dom_segs[0]["start"]
    cur_end = dom_segs[0]["end"]

    for seg in dom_segs[1:]:
        gap_ms = (seg["start"] - cur_end) * 1000
        if gap_ms <= max_gap_ms:
            cur_end = seg["end"]
        else:
            if (cur_end - cur_start) >= min_dur_s:
                merged.append((cur_start, cur_end))
            cur_start = seg["start"]
            cur_end = seg["end"]

    if (cur_end - cur_start) >= min_dur_s:
        merged.append((cur_start, cur_end))

    return merged


def _split_at_silence(
    wav_path: str,
    start_s: float,
    end_s: float,
    max_dur_s: float,
    silence_thresh_dbfs: int,
    min_silence_len_ms: int,
    min_dur_s: float,
) -> list[tuple[float, float]]:
    """
    If a merged segment is too long, split it at silence boundaries.
    Returns list of (start_s, end_s) sub-segments.
    """
    dur_s = end_s - start_s
    if dur_s <= max_dur_s:
        return [(start_s, end_s)]

    # Load the slice
    audio = AudioSegment.from_wav(wav_path)
    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000)
    chunk = audio[start_ms:end_ms]

    # Find silence boundaries within chunk
    silences = detect_silence(
        chunk,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_dbfs,
    )

    if not silences:
        # No silence found — split at target duration
        result = []
        t = start_s
        target = max_dur_s * 0.8  # slightly under max
        while t < end_s - min_dur_s:
            seg_end = min(t + target, end_s)
            result.append((t, seg_end))
            t = seg_end
        return result

    # Split at silence midpoints that are close to multiples of target_duration
    split_points = [start_s]
    for sil_start_ms, sil_end_ms in silences:
        mid_ms = (sil_start_ms + sil_end_ms) // 2
        split_s = start_s + mid_ms / 1000.0
        # Only add if it creates segments of acceptable length
        if split_s - split_points[-1] >= min_dur_s:
            split_points.append(split_s)

    split_points.append(end_s)

    # Build sub-segments
    result = []
    for i in range(len(split_points) - 1):
        seg_start = split_points[i]
        seg_end = split_points[i + 1]
        dur = seg_end - seg_start
        if dur >= min_dur_s and dur <= max_dur_s * 1.1:
            result.append((seg_start, seg_end))

    return result if result else [(start_s, end_s)]


# ── Main segmentation function ────────────────────────────────────────────────

def segment_video(
    diarization: dict,
    config: dict | None = None,
    force: bool = False,
) -> list[dict]:
    """
    Segment a video into clean, single-speaker audio chunks.

    Returns list of segment metadata dicts.
    """
    cfg = config or load_config()
    seg_cfg = cfg["segmentation"]
    audio_cfg = cfg["audio"]
    seg_dir = get_path("segments", cfg)
    meta_dir = get_path("metadata", cfg)
    seg_dir.mkdir(parents=True, exist_ok=True)

    vid_id = diarization["video_id"]
    lang_code = diarization["language_code"]
    wav_path = diarization["wav_path"]
    dominant_speaker = diarization["dominant_speaker"]
    raw_segments = diarization["segments"]

    out_manifest_path = meta_dir / f"{vid_id}_segments.json"
    if not force and out_manifest_path.exists():
        logger.info(f"[skip] Segments already created: {vid_id}")
        return load_json(out_manifest_path)

    logger.info(
        f"[segment] {vid_id} — dominant speaker: {dominant_speaker} "
        f"({diarization['speaker_stats'].get(dominant_speaker, 0):.1f}s)"
    )

    # Step 1: Merge segments
    merged = _merge_segments(
        raw_segments,
        dominant_speaker,
        max_gap_ms=seg_cfg["max_merge_gap_ms"],
        min_dur_s=seg_cfg["min_duration_s"],
        max_dur_s=seg_cfg["max_duration_s"],
    )
    logger.info(f"  Merged into {len(merged)} chunks (before split)")

    # Step 2: Split long chunks at silence
    final_intervals: list[tuple[float, float]] = []
    for start_s, end_s in merged:
        splits = _split_at_silence(
            wav_path, start_s, end_s,
            max_dur_s=seg_cfg["max_duration_s"],
            silence_thresh_dbfs=seg_cfg["silence_thresh_dbfs"],
            min_silence_len_ms=seg_cfg["min_silence_len_ms"],
            min_dur_s=seg_cfg["min_duration_s"],
        )
        final_intervals.extend(splits)

    logger.info(f"  Split into {len(final_intervals)} segments")

    # Step 3: Export each segment
    audio = AudioSegment.from_wav(wav_path)
    segment_metas: list[dict] = []
    target_lufs = audio_cfg.get("target_lufs", -23.0)
    fade_ms = seg_cfg["fade_ms"]

    for idx, (start_s, end_s) in enumerate(
        track(final_intervals, description=f"Exporting {vid_id} segments")
    ):
        seg_id = f"{vid_id}_seg{idx:04d}"
        out_wav = seg_dir / f"{seg_id}.wav"

        if not force and out_wav.exists():
            existing_dur = sf.info(str(out_wav)).duration
            segment_metas.append({
                "segment_id": seg_id,
                "video_id": vid_id,
                "language_code": lang_code,
                "speaker": dominant_speaker,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": round(existing_dur, 3),
                "wav_path": str(out_wav),
            })
            continue

        # Extract slice
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        chunk = audio[start_ms:end_ms]

        # Process
        chunk = _trim_silence(chunk, silence_thresh_dbfs=seg_cfg["silence_thresh_dbfs"])
        chunk = _normalize_loudness(chunk, target_lufs=target_lufs)
        chunk = _apply_fade(chunk, fade_ms=fade_ms)

        # Ensure 16kHz mono
        chunk = chunk.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        # Export
        chunk.export(str(out_wav), format="wav")

        actual_dur = len(chunk) / 1000.0
        segment_metas.append({
            "segment_id": seg_id,
            "video_id": vid_id,
            "language_code": lang_code,
            "speaker": dominant_speaker,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "duration_s": round(actual_dur, 3),
            "wav_path": str(out_wav),
        })

    save_json(segment_metas, out_manifest_path)

    total_dur = sum(s["duration_s"] for s in segment_metas)
    logger.info(
        f"[done] {vid_id}: {len(segment_metas)} segments, "
        f"total={seconds_to_hms(total_dur)}"
    )
    return segment_metas


def segment_all(
    diarization_results: list[dict],
    config: dict | None = None,
    force: bool = False,
) -> list[dict]:
    """Run segmentation on all diarized videos. Returns flat list of all segments."""
    all_segments: list[dict] = []
    for diarization in diarization_results:
        try:
            segs = segment_video(diarization, config, force=force)
            all_segments.extend(segs)
        except Exception as e:
            logger.error(f"Segmentation failed for {diarization.get('video_id')}: {e}")

    total_dur = sum(s["duration_s"] for s in all_segments)
    logger.info(
        f"\n[summary] Total segments: {len(all_segments)}, "
        f"total duration: {seconds_to_hms(total_dur)}"
    )
    return all_segments
