"""
src/transcribe_diarize.py

Sends each chunk to Sarvam's Speech-to-Text API with diarization and
word-level timestamps enabled, and saves a normalized turn list:

    [{"speaker": "SPEAKER_00", "text": "...", "start_sec": 12.3, "end_sec": 18.9}, ...]

This is the file src/segment_by_speaker.py consumes next.

The Sarvam Python SDK API surface has changed across versions (REST vs Batch
vs SDK convenience wrapper) — see https://docs.sarvam.ai/api-reference-docs.
We try the documented high-level SDK call first; if that's unavailable in
your installed SDK version, fall back to the raw batch-job REST flow
(init job -> poll status -> download outputs), which is the lower-level
API guaranteed to exist regardless of SDK convenience-method churn.

Usage:
    export SARVAM_API_KEY=sk_xxx
    python -m src.transcribe_diarize
"""

import csv
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

SARVAM_BASE = "https://api.sarvam.ai"


def _normalize_turns(raw_turns) -> list[dict]:
    out = []
    for t in raw_turns:
        # SDK objects vs plain dicts — handle both
        get = (lambda k, default=None: getattr(t, k, default)) if not isinstance(t, dict) \
            else (lambda k, default=None: t.get(k, default))
        out.append({
            "speaker": get("speaker") or get("speaker_id") or "SPEAKER_00",
            "text": get("text") or get("transcript") or "",
            "start_sec": get("start") or get("start_time") or get("start_sec"),
            "end_sec": get("end") or get("end_time") or get("end_sec"),
        })
    return out


def transcribe_sdk(file_path: Path, language_code: str) -> dict | None:
    try:
        from sarvamai import SarvamAI
    except ImportError:
        return None
    try:
        client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
        response = client.speech_to_text.transcribe(
            file_path=str(file_path),
            language_code=language_code,
            model=config.SARVAM_STT_MODEL,
            with_diarization=True,
            with_timestamps=True,
        )
        raw_turns = getattr(response, "turns", None) or getattr(response, "diarized_transcript", None) or []
        return {
            "transcript_full": getattr(response, "transcript", ""),
            "turns": _normalize_turns(raw_turns),
            "raw": response.dict() if hasattr(response, "dict") else str(response),
        }
    except Exception as e:
        print(f"[sdk path failed, will try REST batch fallback] {e}")
        return None


def transcribe_rest_batch(file_path: Path, language_code: str, poll_every=10, timeout=1800) -> dict | None:
    """Lower-level fallback using the documented batch job lifecycle."""
    headers = {"api-subscription-key": config.SARVAM_API_KEY}

    init = requests.post(
        f"{SARVAM_BASE}/speech-to-text/job/v1",
        headers=headers,
        json={"job_parameters": {
            "language_code": language_code,
            "model": config.SARVAM_STT_MODEL,
            "with_diarization": True,
            "with_timestamps": True,
        }},
    )
    init.raise_for_status()
    job_id = init.json()["job_id"]

    with open(file_path, "rb") as f:
        upload = requests.post(
            f"{SARVAM_BASE}/speech-to-text/job/v1/{job_id}/files",
            headers=headers, files={"file": f},
        )
    upload.raise_for_status()

    requests.post(f"{SARVAM_BASE}/speech-to-text/job/v1/{job_id}/start", headers=headers).raise_for_status()

    elapsed = 0
    while elapsed < timeout:
        status = requests.get(f"{SARVAM_BASE}/speech-to-text/job/v1/{job_id}/status", headers=headers)
        status.raise_for_status()
        state = status.json().get("job_state")
        if state == "Completed":
            break
        if state == "Failed":
            print(f"[error] batch job {job_id} failed")
            return None
        time.sleep(poll_every)
        elapsed += poll_every
    else:
        print(f"[error] batch job {job_id} timed out")
        return None

    dl = requests.post(f"{SARVAM_BASE}/speech-to-text/job/v1/download-files", headers=headers, json={"job_id": job_id})
    dl.raise_for_status()
    result = dl.json()
    output = result["job_details"][0]["outputs"][0]
    raw_turns = output.get("diarized_transcript", {}).get("entries", [])
    return {
        "transcript_full": output.get("transcript", ""),
        "turns": _normalize_turns(raw_turns),
        "raw": output,
    }


def transcribe_chunk(file_path: Path, language_code: str) -> dict | None:
    result = transcribe_sdk(file_path, language_code)
    if result is None:
        result = transcribe_rest_batch(file_path, language_code)
    return result


def main():
    chunk_manifest = config.CHUNKS_DIR / "chunk_manifest.csv"
    if not chunk_manifest.exists():
        print("Run src/preprocess.py first.")
        return
    if not config.SARVAM_API_KEY:
        print("Set SARVAM_API_KEY in your environment first.")
        return

    asr_manifest_path = config.ASR_DIR / "asr_manifest.csv"
    with open(chunk_manifest, newline="", encoding="utf-8") as in_f, \
         open(asr_manifest_path, "w", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.writer(out_f)
        writer.writerow(["source_url", "language", "license", "license_proof", "chunk_path", "asr_json_path"])

        for row in reader:
            chunk_path = Path(row["chunk_path"])
            lang = row["language"]
            print(f"\n=== Transcribing+diarizing {chunk_path.name} [{lang}] ===")

            result = transcribe_chunk(chunk_path, lang)
            if result is None:
                print(f"[skip] could not transcribe {chunk_path}")
                continue

            asr_json_path = config.ASR_DIR / lang / f"{chunk_path.stem}.json"
            asr_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(asr_json_path, "w", encoding="utf-8") as jf:
                json.dump(result, jf, ensure_ascii=False, indent=2)

            writer.writerow([row["source_url"], lang, row["license"], row["license_proof"],
                              str(chunk_path), str(asr_json_path)])
            print(f"-> {len(result['turns'])} turns -> {asr_json_path}")

    print(f"\nDone. Manifest written to {asr_manifest_path}")


if __name__ == "__main__":
    main()
