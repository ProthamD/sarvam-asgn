"""
run_pipeline.py — Main CLI entrypoint for the TTS dataset pipeline.

Usage:
  python run_pipeline.py --stage all
  python run_pipeline.py --stage download
  python run_pipeline.py --stage diarize
  python run_pipeline.py --stage segment
  python run_pipeline.py --stage filter
  python run_pipeline.py --stage transcribe
  python run_pipeline.py --stage tag
  python run_pipeline.py --stage build
  python run_pipeline.py --stage build --dry-run

  python run_pipeline.py --stage all --force   # re-run all stages from scratch
  python run_pipeline.py --add-urls "https://..." --stage download
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pipeline.utils import (
    load_config, get_path, get_env, ensure_dirs,
    save_json, load_json, seconds_to_hms, get_logger
)

console = Console()
logger = get_logger("pipeline")


# ── Stage runner functions ────────────────────────────────────────────────────

def run_download(cfg: dict, force: bool = False) -> list[dict]:
    from pipeline.downloader import download_all
    console.print(Panel("[bold cyan]STAGE 1: Download[/bold cyan]", expand=False))
    return download_all(cfg, force=force)


def run_diarize(cfg: dict, video_metas: list[dict] | None, force: bool = False) -> list[dict]:
    from pipeline.diarizer import diarize_all
    from pipeline.utils import load_json
    console.print(Panel("[bold cyan]STAGE 2: Diarization[/bold cyan]", expand=False))

    if video_metas is None:
        # Load from manifest
        manifest_path = get_path("raw_audio", cfg) / "download_manifest.json"
        if not manifest_path.exists():
            logger.error("No download manifest found. Run --stage download first.")
            sys.exit(1)
        manifest = load_json(manifest_path)
        video_metas = list(manifest.values())

    return diarize_all(video_metas, cfg, force=force)


def run_segment(cfg: dict, diarization_results: list[dict] | None, force: bool = False) -> list[dict]:
    from pipeline.segmenter import segment_all
    console.print(Panel("[bold cyan]STAGE 3: Segmentation[/bold cyan]", expand=False))

    if diarization_results is None:
        # Load from metadata dir
        meta_dir = get_path("metadata", cfg)
        diarization_results = []
        for f in meta_dir.glob("*_diarization.json"):
            diarization_results.append(load_json(f))
        if not diarization_results:
            logger.error("No diarization results found. Run --stage diarize first.")
            sys.exit(1)

    return segment_all(diarization_results, cfg, force=force)


def run_filter(cfg: dict, segments: list[dict] | None, force: bool = False) -> tuple[list[dict], list[dict]]:
    from pipeline.quality_filter import filter_all
    console.print(Panel("[bold cyan]STAGE 4: Quality Filter[/bold cyan]", expand=False))

    if segments is None:
        # Load from all segment manifests
        meta_dir = get_path("metadata", cfg)
        segments = []
        for f in meta_dir.glob("*_segments.json"):
            segments.extend(load_json(f))
        if not segments:
            logger.error("No segment manifests found. Run --stage segment first.")
            sys.exit(1)

    return filter_all(segments, cfg, force=force)


def run_transcribe(cfg: dict, segments: list[dict] | None, force: bool = False) -> list[dict]:
    from pipeline.transcriber import transcribe_all
    console.print(Panel("[bold cyan]STAGE 5: Transcription[/bold cyan]", expand=False))

    if segments is None:
        passed_manifest = get_path("metadata", cfg) / "passed_segments.json"
        if not passed_manifest.exists():
            logger.error("No filtered segments found. Run --stage filter first.")
            sys.exit(1)
        segments = load_json(passed_manifest)

    return transcribe_all(segments, cfg, force=force)


def run_tag(cfg: dict, segments: list[dict] | None, force: bool = False) -> list[dict]:
    from pipeline.emotion_tagger import tag_all
    console.print(Panel("[bold cyan]STAGE 6: Emotion/Style Tagging[/bold cyan]", expand=False))

    if segments is None:
        done_manifest = get_path("metadata", cfg) / "transcribed_segments.json"
        if not done_manifest.exists():
            logger.error("No transcribed segments found. Run --stage transcribe first.")
            sys.exit(1)
        segments = load_json(done_manifest)

    return tag_all(segments, cfg, force=force)


def run_build(cfg: dict, segments: list[dict] | None, dry_run: bool = False) -> str:
    from pipeline.dataset_builder import build_and_push
    console.print(Panel("[bold cyan]STAGE 7: Build & Push Dataset[/bold cyan]", expand=False))

    if segments is None:
        done_manifest = get_path("metadata", cfg) / "tagged_segments.json"
        if not done_manifest.exists():
            # Fallback to transcribed segments
            done_manifest = get_path("metadata", cfg) / "transcribed_segments.json"
        if not done_manifest.exists():
            logger.error("No tagged/transcribed segments found. Run earlier stages first.")
            sys.exit(1)
        
        base_segments = load_json(done_manifest)
        segments_dict = {s["segment_id"]: s for s in base_segments}
        
        # Merge human review decisions
        reviewed_manifest = get_path("metadata", cfg) / "reviewed.jsonl"
        if reviewed_manifest.exists():
            from pipeline.utils import load_jsonl
            reviewed = load_jsonl(reviewed_manifest)
            logger.info(f"Applying {len(reviewed)} human review decisions...")
            for r in reviewed:
                sid = r.get("segment_id")
                if r.get("review_decision") == "rejected":
                    segments_dict.pop(sid, None)
                elif r.get("review_decision") == "accepted":
                    segments_dict[sid] = r
                    
        segments = list(segments_dict.values())

    return build_and_push(segments, cfg, dry_run=dry_run)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TTS Dataset Pipeline — Sarvam AI + HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["all", "download", "diarize", "segment", "filter", "transcribe", "tag", "build"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stage even if cached results exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dataset locally without pushing to HuggingFace",
    )
    parser.add_argument(
        "--add-urls",
        nargs="+",
        metavar="URL",
        help="Additional YouTube URLs to add to the pipeline",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Language code for --add-urls (e.g. en-IN, hi-IN)",
    )

    args = parser.parse_args()

    # ── Load config ─────────────────────────────────────────────────────────
    from pipeline.utils import load_config
    cfg = load_config(args.config)

    # Handle --add-urls
    if args.add_urls:
        lang = args.lang or "en-IN"
        existing = cfg["sources"].get("extra", {"language_code": lang, "urls": []})
        existing["urls"].extend(args.add_urls)
        cfg["sources"]["extra"] = existing
        logger.info(f"Added {len(args.add_urls)} extra URLs under language {lang}")

    ensure_dirs(cfg)

    # ── Validate env vars ────────────────────────────────────────────────────
    try:
        get_env("SARVAM_API_KEY")
        get_env("HF_TOKEN")
    except EnvironmentError as e:
        console.print(f"[bold red]❌ {e}[/bold red]")
        sys.exit(1)

    # ── Print header ────────────────────────────────────────────────────────
    console.print(Panel(
        Text("TTS Dataset Pipeline", style="bold white on blue", justify="center"),
        subtitle=f"Stage: [bold]{args.stage}[/bold] | Force: {args.force}",
    ))

    start_time = time.time()

    # ── Run stages ───────────────────────────────────────────────────────────
    stage = args.stage
    video_metas = None
    diarization_results = None
    segments = None
    passed_segments = None

    if stage in ("all", "download"):
        video_metas = run_download(cfg, force=args.force)
        if stage == "download":
            sys.exit(0)

    if stage in ("all", "diarize"):
        diarization_results = run_diarize(cfg, video_metas, force=args.force)
        if stage == "diarize":
            sys.exit(0)

    if stage in ("all", "segment"):
        segments = run_segment(cfg, diarization_results, force=args.force)
        if stage == "segment":
            sys.exit(0)

    if stage in ("all", "filter"):
        passed_segments, rejected = run_filter(cfg, segments, force=args.force)
        if stage == "filter":
            sys.exit(0)

    if stage in ("all", "transcribe"):
        passed_segments = run_transcribe(cfg, passed_segments, force=args.force)
        if stage == "transcribe":
            sys.exit(0)

    if stage in ("all", "tag"):
        passed_segments = run_tag(cfg, passed_segments, force=args.force)
        if stage == "tag":
            sys.exit(0)

    if stage in ("all", "build"):
        repo_url = run_build(cfg, passed_segments, dry_run=args.dry_run)

    elapsed = time.time() - start_time
    console.print(Panel(
        f"[bold green]Pipeline complete![/bold green]\n"
        f"Elapsed: {seconds_to_hms(elapsed)}\n"
        f"Dataset: {repo_url if stage in ('all', 'build') else 'Not built yet'}",
        title="Summary",
        expand=False,
    ))


if __name__ == "__main__":
    main()
