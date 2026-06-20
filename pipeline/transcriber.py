"""
pipeline/transcriber.py — Accurate transcription of filtered audio segments.

Strategy:
  - Segments <= 25s: Sarvam REST ASR (synchronous, fast)
  - Segments > 25s: Sarvam Batch ASR (asynchronous)
  - Both paths use saaras:v3 with the correct language_code
  - Results include confidence proxy (transcript length / duration)
  - Post-processing: strip extra whitespace, normalize punctuation
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import requests
from rich.progress import track

from pipeline.utils import (
    get_logger, load_config, get_path, get_env,
    save_json, load_json, append_jsonl, seconds_to_hms
)

logger = get_logger("transcriber")

SARVAM_REST = "https://api.sarvam.ai"
REST_MAX_DURATION_S = 25.0  # Use REST API for segments under this threshold


def _split_wav_for_rest(wav_path: Path, max_dur_s: float = 25.0) -> list[Path]:
    """
    Split a WAV file into chunks <= max_dur_s for REST API compatibility.
    Returns list of chunk paths (in /tmp-style next to original).
    """
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(str(wav_path))
    total_ms = len(audio)
    chunk_ms = int(max_dur_s * 1000)

    if total_ms <= chunk_ms:
        return [wav_path]

    chunks = []
    for i, start in enumerate(range(0, total_ms, chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = wav_path.parent / f"{wav_path.stem}_chunk{i}.wav"
        chunk.export(str(chunk_path), format="wav")
        chunks.append(chunk_path)
    return chunks


# ── REST ASR (synchronous, ≤25s) ─────────────────────────────────────────────

def _transcribe_rest(
    api_key: str,
    wav_path: Path,
    language_code: str,
) -> dict:
    """Call Sarvam REST ASR endpoint. Returns raw API response dict."""
    url = f"{SARVAM_REST}/speech-to-text"
    headers = {"api-subscription-key": api_key}

    with open(wav_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (wav_path.name, f, "audio/wav")},
            data={
                "model": "saaras:v3",
                "language_code": language_code,
            },
            headers=headers,
            timeout=60,
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Sarvam ASR REST error {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json()


# ── Batch ASR (asynchronous, >25s) ───────────────────────────────────────────

def _create_batch_job(api_key: str, language_code: str) -> str:
    url = f"{SARVAM_REST}/speech-to-text-async/job"
    payload = {
        "model": "saaras:v3",
        "language_code": language_code,
        "mode": "transcribe",
        "with_diarization": False,
    }
    headers = {"api-subscription-key": api_key}
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["jobId"]


def _upload_to_batch(api_key: str, job_id: str, wav_path: Path) -> None:
    url = f"{SARVAM_REST}/speech-to-text-async/job/{job_id}/files"
    headers = {"api-subscription-key": api_key}
    with open(wav_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (wav_path.name, f, "audio/wav")},
            headers=headers,
            timeout=120,
        )
    resp.raise_for_status()


def _start_batch(api_key: str, job_id: str) -> None:
    url = f"{SARVAM_REST}/speech-to-text-async/job/{job_id}/start"
    headers = {"api-subscription-key": api_key}
    resp = requests.post(url, headers=headers, timeout=30)
    resp.raise_for_status()


def _poll_batch(
    api_key: str,
    job_id: str,
    poll_interval: int = 10,
    timeout: int = 600,
) -> dict:
    url = f"{SARVAM_REST}/speech-to-text-async/job/{job_id}"
    headers = {"api-subscription-key": api_key}
    elapsed = 0
    while elapsed < timeout:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "").lower()
        if status in ("completed", "done"):
            return data
        if status in ("failed", "error"):
            raise RuntimeError(f"Batch job {job_id} failed: {data}")
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Batch job {job_id} timed out after {timeout}s")


def _get_batch_results(api_key: str, job_id: str) -> list[dict]:
    url = f"{SARVAM_REST}/speech-to-text-async/job/{job_id}/files"
    headers = {"api-subscription-key": api_key}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("files", [])


def _transcribe_batch_group(
    api_key: str,
    segments: list[dict],  # segments with same language_code
    poll_interval: int,
) -> dict[str, str]:
    """
    Submit a group of long segments to batch API.
    Returns {segment_id: transcript} mapping.
    """
    if not segments:
        return {}

    lang_code = segments[0]["language_code"]
    job_id = _create_batch_job(api_key, lang_code)
    logger.info(f"[batch] Job {job_id} for {len(segments)} {lang_code} segments")

    # Map filename → segment_id
    file_to_seg: dict[str, str] = {}
    for seg in segments:
        wav_path = Path(seg.get("filtered_wav_path") or seg["wav_path"])
        _upload_to_batch(api_key, job_id, wav_path)
        file_to_seg[wav_path.name] = seg["segment_id"]

    _start_batch(api_key, job_id)
    status_data = _poll_batch(api_key, job_id, poll_interval=poll_interval)

    # Extract transcripts from result
    results: dict[str, str] = {}
    files_output = _get_batch_results(api_key, job_id)

    for file_data in files_output:
        filename = file_data.get("filename", "")
        transcript = (
            file_data.get("transcript")
            or file_data.get("text")
            or file_data.get("result", {}).get("transcript", "")
        )
        if filename in file_to_seg:
            seg_id = file_to_seg[filename]
            results[seg_id] = transcript

    return results


# ── Text post-processing ──────────────────────────────────────────────────────

def _clean_transcript(text: str) -> str:
    """Clean up common ASR artifacts."""
    if not text:
        return ""
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize common filler/artifacts
    text = re.sub(r"\[.*?\]", "", text)  # remove [inaudible], [music], etc.
    text = re.sub(r"\(.*?\)", "", text)  # remove (laughter), (applause), etc.
    text = re.sub(r" {2,}", " ", text).strip()
    # Fix spacing around punctuation
    text = re.sub(r" ([,.!?;:])", r"\1", text)
    return text


def _quality_score(transcript: str, duration_s: float) -> float:
    """Rough quality score: chars per second (expected ~5–12 for normal speech)."""
    if not transcript or duration_s <= 0:
        return 0.0
    return len(transcript) / duration_s


# ── Main transcription function ───────────────────────────────────────────────

def transcribe_all(
    segments: list[dict],
    config: dict | None = None,
    force: bool = False,
) -> list[dict]:
    """
    Transcribe all passed segments. Returns segments with 'transcript' added.
    """
    cfg = config or load_config()
    meta_dir = get_path("metadata", cfg)
    meta_dir.mkdir(parents=True, exist_ok=True)

    transcripts_path = meta_dir / "transcriptions.jsonl"
    done_manifest = meta_dir / "transcribed_segments.json"

    # Load already-done segments
    done_ids: set[str] = set()
    existing: dict[str, str] = {}
    if not force and done_manifest.exists():
        done_data = load_json(done_manifest)
        for item in done_data:
            done_ids.add(item["segment_id"])
            existing[item["segment_id"]] = item.get("transcript", "")
        logger.info(f"[skip] {len(done_ids)} segments already transcribed")

    api_key = get_env("SARVAM_API_KEY")
    sarvam_cfg = cfg["sarvam"]
    poll_interval = sarvam_cfg.get("batch_poll_interval_s", 15)

    # Separate into REST (short) and Batch (long) groups
    rest_queue: list[dict] = []
    batch_queue_by_lang: dict[str, list[dict]] = {}

    for seg in segments:
        sid = seg["segment_id"]
        if sid in done_ids and not force:
            continue
        dur = seg.get("duration_s", 0)
        if dur <= REST_MAX_DURATION_S:
            rest_queue.append(seg)
        else:
            lang = seg["language_code"]
            batch_queue_by_lang.setdefault(lang, []).append(seg)

    logger.info(
        f"[transcribe] {len(rest_queue)} REST segments, "
        f"{sum(len(v) for v in batch_queue_by_lang.values())} batch segments"
    )

    # ── REST transcription ─────────────────────────────────────────────────
    for seg in track(rest_queue, description="REST transcription"):
        wav_path = Path(seg.get("filtered_wav_path") or seg["wav_path"])
        try:
            result = _transcribe_rest(api_key, wav_path, seg["language_code"])
            transcript = _clean_transcript(result.get("transcript", ""))
        except Exception as e:
            logger.warning(f"REST ASR failed for {seg['segment_id']}: {e}")
            transcript = ""
        seg["transcript"] = transcript
        seg["transcription_method"] = "rest"
        seg["transcript_quality_cps"] = round(_quality_score(transcript, seg["duration_s"]), 2)

    # ── Long segments: chunk + REST ────────────────────────────────────────
    long_segs_flat = [s for segs in batch_queue_by_lang.values() for s in segs]
    for seg in track(long_segs_flat, description="Long segment transcription"):
        wav_path = Path(seg.get("filtered_wav_path") or seg["wav_path"])
        try:
            chunks = _split_wav_for_rest(wav_path, max_dur_s=23.0)
            parts = []
            for chunk_path in chunks:
                result = _transcribe_rest(api_key, chunk_path, seg["language_code"])
                parts.append(result.get("transcript", ""))
                # Clean up temp chunk files
                if chunk_path != wav_path:
                    chunk_path.unlink(missing_ok=True)
            transcript = _clean_transcript(" ".join(parts))
        except Exception as e:
            logger.warning(f"Long segment REST failed for {seg['segment_id']}: {e}")
            transcript = ""
        seg["transcript"] = transcript
        seg["transcription_method"] = "rest_chunked"
        seg["transcript_quality_cps"] = round(_quality_score(transcript, seg["duration_s"]), 2)

    # Merge with existing done segments
    updated_segments: list[dict] = []
    for seg in segments:
        sid = seg["segment_id"]
        if sid in done_ids and not force:
            seg["transcript"] = existing.get(sid, "")
            seg["transcription_method"] = "cached"
        updated_segments.append(seg)

    # ── Save results ───────────────────────────────────────────────────────
    save_json(updated_segments, done_manifest)

    # Also save as JSONL for easy inspection
    if transcripts_path.exists():
        transcripts_path.unlink()
    for seg in updated_segments:
        append_jsonl({
            "segment_id": seg["segment_id"],
            "language_code": seg["language_code"],
            "duration_s": seg["duration_s"],
            "transcript": seg.get("transcript", ""),
            "transcript_quality_cps": seg.get("transcript_quality_cps", 0),
            "transcription_method": seg.get("transcription_method", ""),
        }, transcripts_path)

    # ── Summary ────────────────────────────────────────────────────────────
    with_transcript = [s for s in updated_segments if s.get("transcript")]
    empty = len(updated_segments) - len(with_transcript)
    avg_cps = (
        sum(s.get("transcript_quality_cps", 0) for s in with_transcript)
        / max(len(with_transcript), 1)
    )
    logger.info(
        f"[done] {len(with_transcript)}/{len(updated_segments)} segments transcribed "
        f"(avg {avg_cps:.1f} cps). {empty} empty transcripts."
    )

    return updated_segments
