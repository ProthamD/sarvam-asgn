"""
pipeline/quality_filter.py — Multi-stage audio quality gates.

Each gate independently evaluates a segment and writes a rejection reason.
A segment passes only if ALL gates pass. This creates a full audit trail.

Gates (in order):
  1. Duration check
  2. Signal-to-Noise Ratio (SNR)
  3. Clipping detection
  4. Spectral flatness (music / noise rejection)
  5. Silence ratio
  6. Speech rate plausibility (if transcript available)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf
from rich.progress import track
from rich.table import Table
from rich.console import Console

from pipeline.utils import (
    get_logger, load_config, get_path, save_json, load_json,
    append_jsonl, seconds_to_hms
)

logger = get_logger("quality_filter")
console = Console()


# ── Individual gates ──────────────────────────────────────────────────────────

def _check_duration(duration_s: float, min_s: float, max_s: float) -> tuple[bool, str]:
    if duration_s < min_s:
        return False, f"too_short ({duration_s:.1f}s < {min_s}s)"
    if duration_s > max_s:
        return False, f"too_long ({duration_s:.1f}s > {max_s}s)"
    return True, ""


def _check_snr(y: np.ndarray, sr: int, min_snr_db: float) -> tuple[bool, str, float]:
    """
    Estimate SNR by comparing RMS of speech frames vs. noise floor.
    Uses the bottom 10% of frame energies as noise estimate.
    """
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)   # 10ms hop

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms = rms[rms > 0]  # remove zero-energy frames

    if len(rms) < 10:
        return False, "insufficient_audio_content", 0.0

    # Sort and use bottom 10% as noise, top 50% as signal
    sorted_rms = np.sort(rms)
    noise_rms = np.mean(sorted_rms[:max(1, len(sorted_rms) // 10)])
    signal_rms = np.mean(sorted_rms[len(sorted_rms) // 2:])

    if noise_rms == 0:
        snr_db = 60.0  # very clean
    else:
        snr_db = float(20 * np.log10(signal_rms / noise_rms))

    if snr_db < min_snr_db:
        return False, f"low_snr ({snr_db:.1f}dB < {min_snr_db}dB)", round(snr_db, 2)
    return True, "", round(snr_db, 2)


def _check_clipping(y: np.ndarray, max_ratio: float) -> tuple[bool, str, float]:
    """Detect clipping: samples at or near ±1.0."""
    clipped = np.sum(np.abs(y) >= 0.99)
    ratio = float(clipped / len(y))
    if ratio > max_ratio:
        return False, f"clipping ({ratio:.4f} > {max_ratio})", round(ratio, 6)
    return True, "", round(ratio, 6)


def _check_spectral_flatness(y: np.ndarray, sr: int, max_flatness: float) -> tuple[bool, str, float]:
    """
    High spectral flatness → music or noise (not speech).
    Speech typically has low spectral flatness (0.05–0.12).
    Music/noise has high flatness (>0.15).
    """
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness))
    if mean_flatness > max_flatness:
        return False, f"high_spectral_flatness ({mean_flatness:.3f} > {max_flatness})", round(mean_flatness, 4)
    return True, "", round(mean_flatness, 4)


def _check_silence_ratio(y: np.ndarray, sr: int, max_ratio: float) -> tuple[bool, str, float]:
    """Check that the segment isn't mostly silence."""
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # Threshold: -50 dBFS equivalent
    threshold = 10 ** (-50 / 20)
    silent_frames = np.sum(rms < threshold)
    ratio = float(silent_frames / len(rms))
    if ratio > max_ratio:
        return False, f"too_much_silence ({ratio:.2f} > {max_ratio})", round(ratio, 3)
    return True, "", round(ratio, 3)


def _check_speech_rate(
    transcript: str,
    duration_s: float,
    min_cps: float,
    max_cps: float,
) -> tuple[bool, str, float]:
    """Check chars-per-second as a rough speech rate sanity check."""
    if not transcript or not transcript.strip():
        return True, "", 0.0  # No transcript yet, skip gate
    chars = len(transcript.strip())
    cps = chars / duration_s if duration_s > 0 else 0
    if cps < min_cps:
        return False, f"too_slow ({cps:.2f} cps < {min_cps})", round(cps, 2)
    if cps > max_cps:
        return False, f"too_fast ({cps:.2f} cps > {max_cps})", round(cps, 2)
    return True, "", round(cps, 2)


# ── Main filter function ──────────────────────────────────────────────────────

def filter_segment(
    segment: dict,
    config: dict | None = None,
) -> dict:
    """
    Run all quality gates on a single segment.

    Returns updated segment dict with quality fields added:
      - passed: bool
      - rejection_reasons: list[str]
      - snr_db, clipping_ratio, spectral_flatness, silence_ratio, speech_rate_cps
    """
    cfg = config or load_config()
    qc = cfg["quality"]
    seg_cfg = cfg["segmentation"]

    wav_path = segment["wav_path"]
    duration_s = segment["duration_s"]
    transcript = segment.get("transcript", "")

    result = segment.copy()
    result["rejection_reasons"] = []
    result["quality_metrics"] = {}

    # ── Load audio ────────────────────────────────────────────────────────
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
    except Exception as e:
        result["passed"] = False
        result["rejection_reasons"].append(f"load_error: {e}")
        return result

    # ── Gate 1: Duration ──────────────────────────────────────────────────
    ok, reason = _check_duration(
        duration_s,
        min_s=seg_cfg["min_duration_s"],
        max_s=seg_cfg["max_duration_s"],
    )
    if not ok:
        result["rejection_reasons"].append(reason)

    # ── Gate 2: SNR ────────────────────────────────────────────────────────
    ok, reason, snr = _check_snr(y, sr, qc["min_snr_db"])
    result["quality_metrics"]["snr_db"] = snr
    if not ok:
        result["rejection_reasons"].append(reason)

    # ── Gate 3: Clipping ──────────────────────────────────────────────────
    ok, reason, clip_ratio = _check_clipping(y, qc["max_clipping_ratio"])
    result["quality_metrics"]["clipping_ratio"] = clip_ratio
    if not ok:
        result["rejection_reasons"].append(reason)

    # ── Gate 4: Spectral Flatness ─────────────────────────────────────────
    ok, reason, flatness = _check_spectral_flatness(y, sr, qc["max_spectral_flatness"])
    result["quality_metrics"]["spectral_flatness"] = flatness
    if not ok:
        result["rejection_reasons"].append(reason)

    # ── Gate 5: Silence Ratio ─────────────────────────────────────────────
    ok, reason, silence_ratio = _check_silence_ratio(y, sr, qc["max_silence_ratio"])
    result["quality_metrics"]["silence_ratio"] = silence_ratio
    if not ok:
        result["rejection_reasons"].append(reason)

    # ── Gate 6: Speech Rate (if transcript available) ─────────────────────
    ok, reason, cps = _check_speech_rate(
        transcript, duration_s,
        qc["min_speech_rate_cps"], qc["max_speech_rate_cps"]
    )
    result["quality_metrics"]["speech_rate_cps"] = cps
    if not ok:
        result["rejection_reasons"].append(reason)

    result["passed"] = len(result["rejection_reasons"]) == 0
    return result


def filter_all(
    segments: list[dict],
    config: dict | None = None,
    force: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Apply quality filter to all segments.

    Returns (passed_segments, rejected_segments).
    Copies passed WAVs to filtered/ directory.
    """
    cfg = config or load_config()
    filtered_dir = get_path("filtered", cfg)
    meta_dir = get_path("metadata", cfg)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    report_path = meta_dir / "quality_report.jsonl"
    passed_manifest = meta_dir / "passed_segments.json"

    if not force and passed_manifest.exists():
        logger.info("[skip] Quality filter already applied; loading cached results")
        passed = load_json(passed_manifest)
        all_results = load_jsonl(report_path) if report_path.exists() else passed
        rejected = [r for r in all_results if not r.get("passed", True)]
        return passed, rejected

    passed: list[dict] = []
    rejected: list[dict] = []

    # Clear report file for fresh run
    if report_path.exists():
        report_path.unlink()

    for seg in track(segments, description="Quality filtering segments"):
        result = filter_segment(seg, cfg)

        if result["passed"]:
            # Copy to filtered/
            import shutil
            src = Path(result["wav_path"])
            dst = filtered_dir / src.name
            if not dst.exists() or force:
                shutil.copy2(str(src), str(dst))
            result["filtered_wav_path"] = str(dst)
            passed.append(result)
        else:
            rejected.append(result)

        append_jsonl(result, report_path)

    save_json(passed, passed_manifest)

    # ── Summary ────────────────────────────────────────────────────────────
    total = len(segments)
    pass_count = len(passed)
    fail_count = len(rejected)
    pass_dur = sum(s["duration_s"] for s in passed)

    # Count rejection reasons
    reason_counts: dict[str, int] = {}
    for seg in rejected:
        for r in seg.get("rejection_reasons", []):
            key = r.split(" ")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1

    table = Table(title="Quality Filter Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total segments", str(total))
    table.add_row("Passed", f"{pass_count} ({100*pass_count//total}%)")
    table.add_row("Rejected", f"{fail_count} ({100*fail_count//total}%)")
    table.add_row("Passed duration", seconds_to_hms(pass_dur))
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        table.add_row(f"  ↳ {reason}", str(count))

    console.print(table)

    if pass_dur < cfg["dataset"]["min_acceptable_duration_min"] * 60:
        logger.warning(
            f"⚠️  Only {seconds_to_hms(pass_dur)} passed quality filter. "
            f"Consider lowering thresholds in config.yaml or adding more source videos."
        )

    return passed, rejected


def load_jsonl(path):
    """Local helper to avoid circular import."""
    from pipeline.utils import load_jsonl as _load_jsonl
    return _load_jsonl(path)
