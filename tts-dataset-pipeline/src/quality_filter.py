"""
src/quality_filter.py

Cheap, dependency-light audio QA pass over clip_manifest.csv. This is not a
substitute for listening to a sample of the final dataset yourself (do that
too — see QUALITY_CHECKLIST.md), but it catches the worst offenders
automatically: clipped audio, near-silent/noise-floor clips, and
duration outliers.

Usage:
    python -m src.quality_filter
"""

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

FRAME_MS = 20


def frame_rms(samples: np.ndarray, sr: int, frame_ms: int = FRAME_MS) -> np.ndarray:
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.array([np.sqrt(np.mean(samples ** 2) + 1e-12)])
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)


def estimate_snr_db(path: Path) -> float:
    samples, sr = sf.read(str(path))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    rms = frame_rms(samples, sr)
    noise_floor = np.percentile(rms, 10) + 1e-9
    signal_level = np.percentile(rms, 90) + 1e-9
    return 20 * np.log10(signal_level / noise_floor)


def clipping_ratio(path: Path) -> float:
    samples, _ = sf.read(str(path))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return float(np.mean(np.abs(samples) >= 0.99))


def main():
    clip_manifest = config.CLIPS_DIR / "clip_manifest.csv"
    if not clip_manifest.exists():
        print("Run src/segment_by_speaker.py first.")
        return

    passed_path = config.CLIPS_DIR / "clip_manifest_passed.csv"
    flagged_path = config.CLIPS_DIR / "clip_manifest_flagged.csv"

    n_pass, n_fail = 0, 0
    with open(clip_manifest, newline="", encoding="utf-8") as in_f, \
         open(passed_path, "w", newline="", encoding="utf-8") as pass_f, \
         open(flagged_path, "w", newline="", encoding="utf-8") as fail_f:
        reader = csv.DictReader(in_f)
        fieldnames = reader.fieldnames + ["snr_db", "clipping_ratio"]
        pass_writer = csv.DictWriter(pass_f, fieldnames=fieldnames)
        fail_writer = csv.DictWriter(fail_f, fieldnames=fieldnames + ["reject_reason"])
        pass_writer.writeheader()
        fail_writer.writeheader()

        for row in reader:
            clip_path = Path(row["clip_path"])
            duration = float(row["duration_sec"])
            reasons = []

            if not clip_path.exists():
                reasons.append("missing_file")
                snr, clip_ratio = None, None
            else:
                snr = round(estimate_snr_db(clip_path), 1)
                clip_ratio = round(clipping_ratio(clip_path), 5)
                if snr < config.MIN_SNR_DB:
                    reasons.append(f"low_snr({snr}dB)")
                if clip_ratio > config.MAX_CLIPPING_RATIO:
                    reasons.append(f"clipping({clip_ratio})")
                if duration < config.MIN_CLIP_SECONDS:
                    reasons.append("too_short")
                if duration > config.MAX_CLIP_SECONDS:
                    reasons.append("too_long")

            row["snr_db"] = snr
            row["clipping_ratio"] = clip_ratio

            if reasons:
                row["reject_reason"] = ";".join(reasons)
                fail_writer.writerow(row)
                n_fail += 1
            else:
                pass_writer.writerow(row)
                n_pass += 1

    print(f"Passed: {n_pass}  Flagged: {n_fail}")
    print(f"-> {passed_path}")
    print(f"-> {flagged_path}  (spot-check a sample of these before discarding)")
    print("Next: python -m src.emotion_tag")


if __name__ == "__main__":
    main()
