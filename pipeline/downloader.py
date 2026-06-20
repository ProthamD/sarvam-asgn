"""
pipeline/downloader.py — Download YouTube audio via yt-dlp and convert to
16 kHz mono WAV (the canonical TTS format).

Each video is downloaded once; subsequent runs skip already-downloaded files
using a hash manifest stored alongside the raw audio.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import yt_dlp
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from pipeline.utils import (
    get_logger, load_config, get_path, save_json, load_json, seconds_to_hms
)

logger = get_logger("downloader")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract video ID from: {url}")


def _manifest_path(raw_dir: Path) -> Path:
    return raw_dir / "download_manifest.json"


def _load_manifest(raw_dir: Path) -> dict:
    p = _manifest_path(raw_dir)
    return load_json(p) if p.exists() else {}


def _save_manifest(raw_dir: Path, manifest: dict) -> None:
    save_json(manifest, _manifest_path(raw_dir))


# ── Core download function ────────────────────────────────────────────────────

def download_audio(
    url: str,
    language_code: str,
    raw_dir: Path,
    force: bool = False,
) -> Optional[dict]:
    """
    Download a YouTube video's audio and convert to 16 kHz mono WAV.

    Returns a metadata dict or None if already downloaded and force=False.
    """
    vid_id = _video_id(url)
    manifest = _load_manifest(raw_dir)

    out_wav = raw_dir / f"{vid_id}.wav"
    meta_json = raw_dir / f"{vid_id}_meta.json"

    if not force and vid_id in manifest and out_wav.exists():
        logger.info(f"[skip] Already downloaded: {vid_id}")
        return load_json(meta_json) if meta_json.exists() else manifest[vid_id]

    logger.info(f"[download] {url}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── yt-dlp options ──────────────────────────────────────────────────────
    tmp_template = str(raw_dir / f"{vid_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_template,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # Extract info without downloading for metadata
        "writeinfojson": False,
    }

    meta: dict = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info:
            meta = {
                "video_id": vid_id,
                "url": url,
                "language_code": language_code,
                "title": info.get("title", ""),
                "uploader": info.get("uploader", ""),
                "duration_s": info.get("duration", 0),
                "upload_date": info.get("upload_date", ""),
                "view_count": info.get("view_count", 0),
                "description": (info.get("description", "") or "")[:500],
                "thumbnail": info.get("thumbnail", ""),
            }

    # ── Convert to 16 kHz mono WAV via ffmpeg ──────────────────────────────
    # yt-dlp already made a WAV; we resample to ensure correct format
    raw_wav = raw_dir / f"{vid_id}.wav"
    resampled_wav = raw_dir / f"{vid_id}_16k.wav"

    if raw_wav.exists():
        logger.info(f"[ffmpeg] Resampling to 16kHz mono → {resampled_wav.name}")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_wav),
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            str(resampled_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            raise RuntimeError(f"ffmpeg resampling failed for {vid_id}")

        # Replace original with resampled
        raw_wav.unlink()
        resampled_wav.rename(out_wav)
    else:
        logger.warning(f"Expected WAV not found at {raw_wav}; checking for other formats...")
        # Fallback: look for any audio file and convert
        for ext in ["webm", "opus", "m4a", "mp3", "ogg"]:
            alt = raw_dir / f"{vid_id}.{ext}"
            if alt.exists():
                cmd = [
                    "ffmpeg", "-y", "-i", str(alt),
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                    str(out_wav),
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                alt.unlink()
                break

    if not out_wav.exists():
        raise FileNotFoundError(f"Download/conversion failed for {vid_id}")

    # ── Store metadata ──────────────────────────────────────────────────────
    meta["wav_path"] = str(out_wav)
    meta["wav_size_mb"] = round(out_wav.stat().st_size / 1e6, 2)
    if meta.get("duration_s"):
        meta["duration_hms"] = seconds_to_hms(meta["duration_s"])

    save_json(meta, meta_json)
    manifest[vid_id] = meta
    _save_manifest(raw_dir, manifest)

    logger.info(
        f"[done] {meta.get('title', vid_id)[:50]} "
        f"({seconds_to_hms(meta.get('duration_s', 0))})"
    )
    return meta


# ── Batch downloader ─────────────────────────────────────────────────────────

def download_all(config: dict | None = None, force: bool = False) -> list[dict]:
    """Download all configured YouTube sources. Returns list of metadata dicts."""
    cfg = config or load_config()
    raw_dir = get_path("raw_audio", cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_meta: list[dict] = []
    sources = cfg["sources"]

    for lang_key, lang_cfg in sources.items():
        lang_code = lang_cfg["language_code"]
        urls = lang_cfg["urls"]
        logger.info(f"\n{'='*60}")
        logger.info(f"  Language: {lang_code} ({len(urls)} videos)")
        logger.info(f"{'='*60}")

        for url in urls:
            try:
                meta = download_audio(url, lang_code, raw_dir, force=force)
                if meta:
                    all_meta.append(meta)
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")

    total_s = sum(m.get("duration_s", 0) for m in all_meta)
    logger.info(
        f"\n[summary] Downloaded {len(all_meta)} videos, "
        f"total raw duration: {seconds_to_hms(total_s)}"
    )
    return all_meta
