"""
src/preprocess.py

For every file in raw_manifest.csv:
  1. Loudness-normalize (ffmpeg loudnorm, EBU R128) so volume is consistent
     across all sources before we ever look at SNR.
  2. Split into chunks no longer than config.SARVAM_BATCH_MAX_SECONDS, since
     that's what we'll hand to the Sarvam batch STT+diarization call.

Usage:
    python -m src.preprocess
"""

import csv
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def get_duration_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def loudnorm(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-af", "loudnorm=I=-23:LRA=7:TP=-2",
        "-ar", str(config.ASR_SAMPLE_RATE), "-ac", "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def split_into_chunks(src: Path, out_dir: Path, max_seconds: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = get_duration_seconds(src)
    n_chunks = max(1, math.ceil(duration / max_seconds))
    chunk_len = duration / n_chunks
    chunk_paths = []
    for i in range(n_chunks):
        start = i * chunk_len
        chunk_path = out_dir / f"{src.stem}__chunk{i:02d}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-ss", str(start), "-t", str(chunk_len),
            "-ar", str(config.ASR_SAMPLE_RATE), "-ac", "1",
            str(chunk_path),
        ]
        subprocess.run(cmd, check=True)
        chunk_paths.append(chunk_path)
    return chunk_paths


def main():
    raw_manifest = config.RAW_DIR / "raw_manifest.csv"
    if not raw_manifest.exists():
        print("Run src/download.py first.")
        return

    chunk_manifest_path = config.CHUNKS_DIR / "chunk_manifest.csv"
    with open(raw_manifest, newline="", encoding="utf-8") as in_f, \
         open(chunk_manifest_path, "w", newline="", encoding="utf-8") as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.writer(out_f)
        writer.writerow(["source_url", "language", "license", "license_proof", "chunk_path"])

        for row in reader:
            raw_path = Path(row["raw_path"])
            lang = row["language"]
            print(f"\n=== Preprocessing {raw_path.name} [{lang}] ===")

            normed_path = config.CHUNKS_DIR / "_normed" / lang / raw_path.name
            loudnorm(raw_path, normed_path)

            chunk_out_dir = config.CHUNKS_DIR / lang
            chunks = split_into_chunks(normed_path, chunk_out_dir, config.SARVAM_BATCH_MAX_SECONDS)

            for chunk_path in chunks:
                writer.writerow([row["source_url"], lang, row["license"], row["license_proof"], str(chunk_path)])
                print(f"-> {chunk_path}")

    print(f"\nDone. Manifest written to {chunk_manifest_path}")


if __name__ == "__main__":
    main()
