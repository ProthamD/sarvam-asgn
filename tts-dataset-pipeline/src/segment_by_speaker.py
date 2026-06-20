"""
src/segment_by_speaker.py

Consumes asr_manifest.csv (chunk audio + normalized diarization turns) and:
  1. Groups consecutive same-speaker turns, merging them up to
     config.TARGET_CLIP_SECONDS (capped at MAX_CLIP_SECONDS).
  2. Drops anything under MIN_CLIP_SECONDS or where timestamps are missing
     (those get written to data/review/ instead, for manual cutting).
  3. Cuts the actual single-speaker WAV clip out of the source chunk with
     ffmpeg and writes a flat manifest: clip_path, text, speaker_id,
     language, duration_sec, source_url, license.

Usage:
    python -m src.segment_by_speaker
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def merge_turns(turns: list[dict]) -> list[dict]:
    """Greedy merge of consecutive same-speaker turns toward TARGET_CLIP_SECONDS."""
    merged = []
    buf = None

    def flush():
        nonlocal buf
        if buf is not None:
            dur = buf["end_sec"] - buf["start_sec"]
            if dur >= config.MIN_CLIP_SECONDS:
                merged.append(buf)
        buf = None

    for t in turns:
        if t["start_sec"] is None or t["end_sec"] is None or not t["text"].strip():
            continue
        if buf is None:
            buf = dict(t)
            continue
        same_speaker = t["speaker"] == buf["speaker"]
        gap = t["start_sec"] - buf["end_sec"]
        current_dur = buf["end_sec"] - buf["start_sec"]
        would_fit = current_dur + (t["end_sec"] - t["start_sec"]) <= config.MAX_CLIP_SECONDS
        if same_speaker and gap < 1.5 and would_fit and current_dur < config.TARGET_CLIP_SECONDS:
            buf["end_sec"] = t["end_sec"]
            buf["text"] = (buf["text"].rstrip() + " " + t["text"].strip()).strip()
        else:
            flush()
            buf = dict(t)
    flush()
    return merged


def cut_clip(chunk_path: Path, start_sec: float, end_sec: float, out_path: Path):
    pad = config.SILENCE_PAD_MS / 1000.0
    start = max(0.0, start_sec - pad)
    duration = (end_sec - start_sec) + 2 * pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(chunk_path),
        "-ss", str(start), "-t", str(duration),
        "-ar", str(config.TARGET_SAMPLE_RATE), "-ac", "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    asr_manifest = config.ASR_DIR / "asr_manifest.csv"
    if not asr_manifest.exists():
        print("Run src/transcribe_diarize.py first.")
        return

    clip_manifest_path = config.CLIPS_DIR / "clip_manifest.csv"
    clip_counter = {}  # per-language running index for clean filenames

    with open(asr_manifest, newline="", encoding="utf-8") as in_f, \
         open(clip_manifest_path, "w", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.writer(out_f)
        writer.writerow(["clip_path", "text", "speaker_id", "language",
                          "duration_sec", "source_url", "license", "source_chunk"])

        for row in reader:
            lang = row["language"]
            chunk_path = Path(row["chunk_path"])
            asr_json_path = Path(row["asr_json_path"])
            with open(asr_json_path, encoding="utf-8") as jf:
                asr_data = json.load(jf)

            turns = sorted(
                (t for t in asr_data["turns"] if t["start_sec"] is not None),
                key=lambda t: t["start_sec"],
            )
            if not turns:
                review_path = config.REVIEW_DIR / asr_json_path.name
                review_path.parent.mkdir(parents=True, exist_ok=True)
                review_path.write_text(json.dumps(asr_data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[review] no usable timestamps for {chunk_path.name}, copied to {review_path}")
                continue

            clips = merge_turns(turns)
            print(f"\n=== {chunk_path.name} [{lang}]: {len(turns)} turns -> {len(clips)} candidate clips ===")

            clip_counter.setdefault(lang, 0)
            for c in clips:
                clip_counter[lang] += 1
                idx = clip_counter[lang]
                source_tag = chunk_path.stem.split("__chunk")[0]
                speaker_id = f"{source_tag}__{c['speaker']}"
                clip_name = f"{lang}_{idx:05d}.wav"
                clip_path = config.CLIPS_DIR / lang / clip_name

                cut_clip(chunk_path, c["start_sec"], c["end_sec"], clip_path)
                duration = round(c["end_sec"] - c["start_sec"], 2)

                writer.writerow([str(clip_path), c["text"].strip(), speaker_id, lang,
                                  duration, row["source_url"], row["license"], str(chunk_path)])

    print(f"\nDone. Clip manifest written to {clip_manifest_path}")
    print("Next: python -m src.quality_filter")


if __name__ == "__main__":
    main()
