"""
pipeline/diarizer.py — Speaker diarization via Sarvam Batch ASR API.

Strategy:
  1. Submit the full downloaded WAV to Sarvam's batch job with diarization=True.
  2. Poll until complete, download output JSON.
  3. Parse speaker-labeled segments and identify the dominant speaker (most
     cumulative speaking time) to keep for single-speaker TTS data.
  4. Save diarization result per video for use by the segmenter.

Sarvam batch API supports files up to 2 hours and returns speaker timestamps
+ transcripts when diarization is enabled.
"""
from __future__ import annotations

import os
import time
import json
import tempfile
from pathlib import Path
from typing import Optional

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn

from pipeline.utils import (
    get_logger, load_config, get_path, get_env,
    save_json, load_json, seconds_to_hms, append_jsonl
)

logger = get_logger("diarizer")

SARVAM_REST = "https://api.sarvam.ai"


# ── REST helpers ─────────────────────────────────────────────────────────────

def _headers(api_key: str) -> dict:
    return {"api-subscription-key": api_key}


def _create_batch_job(api_key: str, language_code: str, with_diarization: bool, num_speakers: Optional[int]) -> str:
    """Create a batch ASR job and return job_id."""
    url = f"{SARVAM_REST}/speech-to-text-async/job"
    payload = {
        "model": "saaras:v3",
        "language_code": language_code,
        "mode": "transcribe",
        "with_diarization": with_diarization,
    }
    if num_speakers:
        payload["num_speakers"] = num_speakers

    resp = requests.post(url, json=payload, headers=_headers(api_key), timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["jobId"]
    logger.info(f"[batch] Created job: {job_id}")
    return job_id


def _upload_file_to_job(api_key: str, job_id: str, wav_path: Path) -> None:
    """Upload a WAV file to an existing batch job."""
    url = f"{SARVAM_REST}/speech-to-text-async/job/{job_id}/files"
    with open(wav_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (wav_path.name, f, "audio/wav")},
            headers=_headers(api_key),
            timeout=120,
        )
    resp.raise_for_status()
    logger.info(f"[batch] Uploaded {wav_path.name} to job {job_id}")


def _start_job(api_key: str, job_id: str) -> None:
    url = f"{SARVAM_REST}/speech-to-text-async/job/{job_id}/start"
    resp = requests.post(url, headers=_headers(api_key), timeout=30)
    resp.raise_for_status()
    logger.info(f"[batch] Started job {job_id}")


def _poll_job(api_key: str, job_id: str, poll_interval: int, timeout: int) -> dict:
    """Poll job until complete. Returns job status dict with results."""
    url = f"{SARVAM_REST}/speech-to-text-async/job/{job_id}"
    elapsed = 0
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task(f"Waiting for Sarvam batch job {job_id[:8]}...", total=None)
        while elapsed < timeout:
            resp = requests.get(url, headers=_headers(api_key), timeout=30)
            resp.raise_for_status()
            status_data = resp.json()
            status = status_data.get("status", "")

            if status in ("completed", "COMPLETED", "done", "DONE"):
                progress.update(task, description=f"✅ Job {job_id[:8]} completed")
                return status_data
            elif status in ("failed", "FAILED", "error", "ERROR"):
                raise RuntimeError(f"Batch job {job_id} failed: {status_data}")

            progress.update(task, description=f"Waiting for job {job_id[:8]}... [{elapsed}s elapsed, status={status}]")
            time.sleep(poll_interval)
            elapsed += poll_interval

    raise TimeoutError(f"Batch job {job_id} timed out after {timeout}s")


def _get_job_output(api_key: str, job_id: str) -> list[dict]:
    """Fetch the output transcription for each file in the job."""
    url = f"{SARVAM_REST}/speech-to-text-async/job/{job_id}/files"
    resp = requests.get(url, headers=_headers(api_key), timeout=30)
    resp.raise_for_status()
    return resp.json().get("files", [])


# ── Fallback: REST API direct transcription (no diarization, for short files) ─

def transcribe_rest(
    api_key: str,
    wav_path: Path,
    language_code: str,
) -> dict:
    """
    Call Sarvam REST ASR (for files ≤ 30s, no diarization).
    Returns: {"transcript": str, "language_code": str}
    """
    url = f"{SARVAM_REST}/speech-to-text"
    with open(wav_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (wav_path.name, f, "audio/wav")},
            data={
                "model": "saaras:v3",
                "language_code": language_code,
                "with_timestamps": "false",
            },
            headers=_headers(api_key),
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()


# ── Diarization result parsing ────────────────────────────────────────────────

def _parse_diarization_output(output_data: dict | list) -> list[dict]:
    """
    Parse Sarvam batch job output into a list of speaker segments:
    [{"speaker": str, "start": float, "end": float, "transcript": str}, ...]

    Sarvam's output format can vary; we handle the most common shapes.
    """
    segments: list[dict] = []

    # Handle list of file outputs
    if isinstance(output_data, list):
        for item in output_data:
            segments.extend(_parse_diarization_output(item))
        return segments

    # Handle single file output dict
    if isinstance(output_data, dict):
        # Shape 1: {"diarization": [...]}
        diarization = output_data.get("diarization") or output_data.get("speakers")
        if diarization and isinstance(diarization, list):
            for seg in diarization:
                segments.append({
                    "speaker": seg.get("speaker", seg.get("speaker_id", "SPEAKER_0")),
                    "start": float(seg.get("start", seg.get("start_time", 0))),
                    "end": float(seg.get("end", seg.get("end_time", 0))),
                    "transcript": seg.get("transcript", seg.get("text", "")),
                })
            return segments

        # Shape 2: {"transcript": "...", "words": [...]} (no diarization)
        transcript = output_data.get("transcript", "")
        if transcript:
            segments.append({
                "speaker": "SPEAKER_0",
                "start": 0.0,
                "end": output_data.get("duration", 0.0),
                "transcript": transcript,
            })

    return segments


def _dominant_speaker(segments: list[dict]) -> str:
    """Find the speaker with the most cumulative speaking time."""
    speaker_time: dict[str, float] = {}
    for seg in segments:
        sp = seg["speaker"]
        dur = seg["end"] - seg["start"]
        speaker_time[sp] = speaker_time.get(sp, 0.0) + dur
    if not speaker_time:
        return "SPEAKER_0"
    return max(speaker_time, key=speaker_time.__getitem__)


# ── Main diarize function ─────────────────────────────────────────────────────

def diarize_video(
    video_meta: dict,
    config: dict | None = None,
    force: bool = False,
) -> dict:
    """
    Run speaker diarization on a downloaded video WAV.

    Returns diarization result dict with keys:
      - video_id, language_code, dominant_speaker
      - segments: [{"speaker", "start", "end", "transcript"}]
      - speaker_stats: {speaker_id: total_seconds}
    """
    cfg = config or load_config()
    meta_dir = get_path("metadata", cfg)
    meta_dir.mkdir(parents=True, exist_ok=True)

    vid_id = video_meta["video_id"]
    lang_code = video_meta["language_code"]
    wav_path = Path(video_meta["wav_path"])

    out_path = meta_dir / f"{vid_id}_diarization.json"

    if not force and out_path.exists():
        logger.info(f"[skip] Diarization already done: {vid_id}")
        return load_json(out_path)

    api_key = get_env("SARVAM_API_KEY")
    sarvam_cfg = cfg["sarvam"]

    logger.info(f"[diarize] {vid_id} ({lang_code}) — {wav_path.name}")

    try:
        # ── Batch API with diarization ─────────────────────────────────────
        num_speakers = sarvam_cfg.get("num_speakers")
        job_id = _create_batch_job(
            api_key, lang_code,
            with_diarization=sarvam_cfg["with_diarization"],
            num_speakers=num_speakers,
        )
        _upload_file_to_job(api_key, job_id, wav_path)
        _start_job(api_key, job_id)

        status_data = _poll_job(
            api_key, job_id,
            poll_interval=sarvam_cfg["batch_poll_interval_s"],
            timeout=sarvam_cfg["batch_timeout_s"],
        )

        # Get file outputs
        files_output = _get_job_output(api_key, job_id)
        
        # Try to get output from status data or files
        raw_output = files_output or status_data
        segments = _parse_diarization_output(raw_output)

    except Exception as e:
        logger.warning(
            f"[diarize] Batch API failed for {vid_id}: {e}\n"
            f"  Falling back to no-diarization (single speaker assumed)."
        )
        # Fallback: treat entire file as single speaker
        import soundfile as sf
        info = sf.info(str(wav_path))
        segments = [{
            "speaker": "SPEAKER_0",
            "start": 0.0,
            "end": info.duration,
            "transcript": "",  # Will be transcribed later
        }]

    if not segments:
        logger.warning(f"[diarize] No segments found for {vid_id}, using full audio")
        import soundfile as sf
        info = sf.info(str(wav_path))
        segments = [{"speaker": "SPEAKER_0", "start": 0.0, "end": info.duration, "transcript": ""}]

    # ── Speaker stats ──────────────────────────────────────────────────────
    speaker_stats: dict[str, float] = {}
    for seg in segments:
        sp = seg["speaker"]
        dur = seg["end"] - seg["start"]
        speaker_stats[sp] = speaker_stats.get(sp, 0.0) + dur

    dominant = _dominant_speaker(segments)

    result = {
        "video_id": vid_id,
        "language_code": lang_code,
        "wav_path": str(wav_path),
        "dominant_speaker": dominant,
        "num_speakers_detected": len(speaker_stats),
        "speaker_stats": {k: round(v, 2) for k, v in speaker_stats.items()},
        "total_speech_s": round(sum(speaker_stats.values()), 2),
        "segments": segments,
    }

    save_json(result, out_path)
    logger.info(
        f"[done] {vid_id}: {len(segments)} segments, "
        f"{len(speaker_stats)} speakers, dominant={dominant} "
        f"({seconds_to_hms(speaker_stats.get(dominant, 0))})"
    )
    return result


def diarize_all(
    video_metas: list[dict],
    config: dict | None = None,
    force: bool = False,
) -> list[dict]:
    """Run diarization on all downloaded videos."""
    results = []
    for meta in video_metas:
        try:
            result = diarize_video(meta, config, force=force)
            results.append(result)
        except Exception as e:
            logger.error(f"Diarization failed for {meta.get('video_id')}: {e}")
    return results
