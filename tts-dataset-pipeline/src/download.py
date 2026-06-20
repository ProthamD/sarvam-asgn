"""
src/download.py

Reads sources.csv (url, language, license, license_proof, notes) and downloads
the best-quality audio track for each entry using yt-dlp, converting to mono
WAV at config.ASR_SAMPLE_RATE. Skips rows without a recognised license value
so an unverified source can never silently enter the pipeline.

Usage:
    python -m src.download
"""

import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_sources(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("url") or row["url"].strip().startswith("#"):
                continue
            if row.get("language") not in config.LANGUAGES:
                print(f"[skip] unknown language '{row.get('language')}' for {row['url']}")
                continue
            if row.get("license") not in config.ALLOWED_LICENSES:
                print(f"[skip] missing/invalid license for {row['url']} "
                      f"(got '{row.get('license')}', must be one of {config.ALLOWED_LICENSES})")
                continue
            if not row.get("license_proof"):
                print(f"[skip] no license_proof recorded for {row['url']}")
                continue
            rows.append(row)
    return rows


def download_one(url: str, language: str, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", out_template,
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] yt-dlp failed for {url}: {e}")
        return None

    # find the file yt-dlp just produced
    wavs = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return wavs[0] if wavs else None


def to_pipeline_format(src_wav: Path, dst_wav: Path):
    """Mono, ASR_SAMPLE_RATE, 16-bit PCM — what Sarvam STT expects."""
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src_wav),
        "-ac", "1", "-ar", str(config.ASR_SAMPLE_RATE),
        "-sample_fmt", "s16",
        str(dst_wav),
    ]
    subprocess.run(cmd, check=True)


def main():
    rows = load_sources(config.ROOT / "sources.csv")
    if not rows:
        print("No valid rows in sources.csv yet — see SOURCING_GUIDE.md, then fill it in.")
        return

    manifest_path = config.RAW_DIR / "raw_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["source_url", "language", "license", "license_proof", "raw_path"])

        for row in rows:
            lang = row["language"]
            print(f"\n=== Downloading [{lang}] {row['url']} ===")
            tmp_dir = config.RAW_DIR / "_tmp"
            raw_file = download_one(row["url"], lang, tmp_dir)
            if raw_file is None:
                continue

            final_name = f"{lang}__{raw_file.stem}.wav"
            final_path = config.RAW_DIR / lang / final_name
            to_pipeline_format(raw_file, final_path)
            raw_file.unlink(missing_ok=True)

            writer.writerow([row["url"], lang, row["license"], row["license_proof"], str(final_path)])
            print(f"-> saved {final_path}")

    print(f"\nDone. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
