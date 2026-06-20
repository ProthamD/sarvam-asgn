"""
review_tool.py — Interactive CLI review tool.

Listen to each filtered segment, see its transcription and emotion tag,
then accept, reject, or manually correct the transcription/tag.

Controls:
  [a] Accept segment as-is
  [r] Reject segment (with optional reason)
  [e] Edit transcription
  [t] Edit emotion/style tags
  [p] Re-play audio
  [s] Skip (decide later)
  [q] Quit and save progress

Results saved to data/metadata/reviewed.jsonl
Resume-safe: already-reviewed segments are skipped.
"""
from __future__ import annotations

import json
import os
import sys
import platform
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text

from pipeline.utils import (
    load_config, get_path, get_env,
    load_json, save_json, append_jsonl, load_jsonl, seconds_to_hms,
    get_logger
)

console = Console()
logger = get_logger("review_tool")


# ── Audio playback ────────────────────────────────────────────────────────────

def _play_audio(wav_path: str) -> None:
    """Play audio file using platform-appropriate method."""
    try:
        if platform.system() == "Windows":
            # Use Windows Media Player or PowerShell
            subprocess.Popen(
                ["powershell", "-c", f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).wait()
        elif platform.system() == "Darwin":
            subprocess.Popen(["afplay", wav_path]).wait()
        else:
            subprocess.Popen(["aplay", wav_path]).wait()
    except Exception:
        try:
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(wav_path)
            sd.play(data, sr)
            sd.wait()
        except Exception as e2:
            console.print(f"[yellow]⚠ Cannot play audio: {e2}[/yellow]")


# ── Display helpers ───────────────────────────────────────────────────────────

def _show_segment(seg: dict, idx: int, total: int) -> None:
    """Display segment info panel."""
    lang_display = "🇮🇳 Hindi" if "hi" in seg.get("language_code", "") else "🇬🇧 Indian English"
    snr = seg.get("quality_metrics", {}).get("snr_db", 0)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="dim cyan", width=22)
    table.add_column("Value", style="white")

    table.add_row("Segment ID", seg["segment_id"])
    table.add_row("Language", lang_display)
    table.add_row("Duration", f"{seg.get('duration_s', 0):.1f}s")
    table.add_row("SNR", f"{snr:.1f} dB")
    table.add_row("Source", seg.get("video_id", "unknown"))
    table.add_row("Emotion", f"[bold yellow]{seg.get('primary_emotion', '?')}[/bold yellow]")
    table.add_row("Style", f"[bold magenta]{seg.get('speaking_style', '?')}[/bold magenta]")
    table.add_row("Rate", seg.get("speech_rate", "?"))
    table.add_row("Tag confidence", f"{seg.get('tag_confidence', 0):.2f}")
    table.add_row("", "")
    table.add_row("Transcript", f"[bold white]{seg.get('transcript', '[empty]')}[/bold white]")

    console.print(Panel(
        table,
        title=f"[bold cyan]Segment {idx + 1}/{total}[/bold cyan]",
        border_style="blue",
    ))


def _show_controls() -> None:
    console.print(
        "[dim][[a] Accept | [r] Reject | [e] Edit transcript | "
        "[t] Edit tags | [p] Re-play | [s] Skip | [q] Quit][/dim]"
    )


# ── Review loop ───────────────────────────────────────────────────────────────

VALID_EMOTIONS = ["neutral", "happy", "sad", "excited", "angry", "fearful", "surprised", "calm", "serious"]
VALID_STYLES = ["formal", "informal", "narrative", "conversational", "instructional",
                "interview", "debate", "prayer", "news_reading", "storytelling", "whisper"]
VALID_RATES = ["slow", "normal", "fast"]


def review_all(
    segments: list[dict],
    config: dict | None = None,
    sample_rate: float = 1.0,  # 1.0 = review all, 0.1 = random 10%
) -> dict:
    """
    Interactive review of segments. Returns summary dict.
    """
    import random
    cfg = config or load_config()
    meta_dir = get_path("metadata", cfg)
    meta_dir.mkdir(parents=True, exist_ok=True)

    reviewed_path = meta_dir / "reviewed.jsonl"
    reviewed_manifest = meta_dir / "reviewed_segments.json"

    # Load already-reviewed IDs
    already_reviewed: set[str] = set()
    review_decisions: dict[str, dict] = {}
    if reviewed_path.exists():
        for rec in load_jsonl(reviewed_path):
            sid = rec.get("segment_id", "")
            already_reviewed.add(sid)
            review_decisions[sid] = rec

    # Filter and optionally sample
    to_review = [s for s in segments if s["segment_id"] not in already_reviewed]
    if sample_rate < 1.0:
        n = max(1, int(len(to_review) * sample_rate))
        to_review = random.sample(to_review, min(n, len(to_review)))
        console.print(f"[yellow]Sampling {len(to_review)} segments for review[/yellow]")

    console.print(Panel(
        f"[bold white]🎤 TTS Dataset Review Tool[/bold white]\n\n"
        f"Total segments: [cyan]{len(segments)}[/cyan]\n"
        f"Already reviewed: [green]{len(already_reviewed)}[/green]\n"
        f"To review now: [yellow]{len(to_review)}[/yellow]",
        border_style="blue",
    ))

    if not to_review:
        console.print("[green]✅ All segments already reviewed![/green]")
        return {"total_reviewed": len(already_reviewed)}

    accepted = 0
    rejected = 0
    skipped = 0
    edited = 0

    for idx, seg in enumerate(to_review):
        wav_path = seg.get("filtered_wav_path") or seg.get("wav_path", "")

        console.clear()
        _show_segment(seg, idx, len(to_review))

        # Auto-play
        console.print("[dim]Playing audio...[/dim]")
        _play_audio(wav_path)

        while True:
            _show_controls()
            choice = Prompt.ask("Action", choices=["a", "r", "e", "t", "p", "s", "q"], default="a")

            if choice == "p":
                _play_audio(wav_path)
                continue

            elif choice == "a":
                decision = {**seg, "review_decision": "accepted", "review_edited": False}
                append_jsonl(decision, reviewed_path)
                review_decisions[seg["segment_id"]] = decision
                accepted += 1
                break

            elif choice == "r":
                reason = Prompt.ask("Rejection reason (optional)", default="manual_reject")
                decision = {**seg, "review_decision": "rejected", "rejection_reason": reason}
                append_jsonl(decision, reviewed_path)
                review_decisions[seg["segment_id"]] = decision
                rejected += 1
                break

            elif choice == "e":
                console.print(f"[dim]Current: {seg.get('transcript', '')}[/dim]")
                new_transcript = Prompt.ask("New transcript")
                seg["transcript"] = new_transcript
                seg["review_edited"] = True
                edited += 1
                continue  # Stay in loop to allow accept/reject

            elif choice == "t":
                console.print(f"[dim]Emotions: {', '.join(VALID_EMOTIONS)}[/dim]")
                new_emotion = Prompt.ask("Emotion", default=seg.get("primary_emotion", "neutral"))
                if new_emotion in VALID_EMOTIONS:
                    seg["primary_emotion"] = new_emotion

                console.print(f"[dim]Styles: {', '.join(VALID_STYLES)}[/dim]")
                new_style = Prompt.ask("Style", default=seg.get("speaking_style", "conversational"))
                if new_style in VALID_STYLES:
                    seg["speaking_style"] = new_style

                new_rate = Prompt.ask("Rate", choices=VALID_RATES, default=seg.get("speech_rate", "normal"))
                seg["speech_rate"] = new_rate
                seg["review_edited"] = True
                edited += 1
                continue

            elif choice == "s":
                skipped += 1
                break

            elif choice == "q":
                console.print(
                    f"\n[bold]Review paused.[/bold] "
                    f"Accepted: {accepted}, Rejected: {rejected}, Skipped: {skipped}, Edited: {edited}"
                )
                break

        if choice == "q":
            break

    # Save final manifest of all accepted segments
    all_accepted = []
    for sid, decision in review_decisions.items():
        if decision.get("review_decision") == "accepted":
            all_accepted.append(decision)

    save_json(all_accepted, reviewed_manifest)

    summary = {
        "total_reviewed": len(already_reviewed) + accepted + rejected,
        "accepted": accepted,
        "rejected": rejected,
        "skipped": skipped,
        "edited": edited,
        "total_accepted": len(all_accepted),
        "accepted_duration_s": sum(s.get("duration_s", 0) for s in all_accepted),
    }

    console.print(Panel(
        f"[bold green]Review Session Complete[/bold green]\n\n"
        f"Accepted: [green]{accepted}[/green]\n"
        f"Rejected: [red]{rejected}[/red]\n"
        f"Skipped: [yellow]{skipped}[/yellow]\n"
        f"Edited: [cyan]{edited}[/cyan]\n\n"
        f"Total accepted (all sessions): [bold]{len(all_accepted)}[/bold] "
        f"({seconds_to_hms(summary['accepted_duration_s'])})",
        border_style="green",
    ))
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive TTS segment review tool")
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fraction of segments to sample for review (e.g. 0.1 for 10%)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Config file path"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    meta_dir = get_path("metadata", cfg)

    # Load tagged or transcribed segments
    for candidate in ["tagged_segments.json", "transcribed_segments.json", "passed_segments.json"]:
        path = meta_dir / candidate
        if path.exists():
            segments = load_json(path)
            console.print(f"[dim]Loaded {len(segments)} segments from {candidate}[/dim]")
            break
    else:
        console.print("[bold red]❌ No segment data found. Run pipeline stages first.[/bold red]")
        sys.exit(1)

    review_all(segments, cfg, sample_rate=args.sample)


if __name__ == "__main__":
    main()
