"""
src/build_hf_dataset.py

Takes clip_manifest_tagged.csv (optionally hand-corrected) and assembles a
HuggingFace `AudioFolder`-compatible dataset:

    data/final/
      train/
        en-IN/
          en-IN_00001.wav
          ...
        bn-IN/
          ...
        metadata.csv      <- AudioFolder convention: file_name,*
      README.md            <- dataset card

Run with --push to also upload to the Hub (requires `huggingface-cli login`
or HF_TOKEN env var, and a repo id via --repo).

Usage:
    python -m src.build_hf_dataset
    python -m src.build_hf_dataset --push --repo your-username/indian-tts-dataset
"""

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_local_dataset(tagged_csv: Path):
    train_dir = config.FINAL_DIR / "train"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True)

    metadata_rows = []
    duration_by_lang = defaultdict(float)
    count_by_lang = defaultdict(int)
    emotion_counts = defaultdict(int)
    style_counts = defaultdict(int)
    licenses_by_lang = defaultdict(set)
    sources_by_lang = defaultdict(set)

    with open(tagged_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        lang = row["language"]
        src_clip = Path(row["clip_path"])
        if not src_clip.exists():
            print(f"[skip] missing audio file {src_clip}")
            continue

        lang_dir = train_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        dst_clip = lang_dir / src_clip.name
        shutil.copy2(src_clip, dst_clip)

        rel_path = f"{lang}/{src_clip.name}"
        metadata_rows.append({
            "file_name": rel_path,
            "text": row["text"],
            "language": lang,
            "emotion": row["emotion"],
            "style": row["style"],
            "speaker_id": row["speaker_id"],
            "duration_sec": row["duration_sec"],
            "source_url": row["source_url"],
            "license": row["license"],
        })

        duration_by_lang[lang] += float(row["duration_sec"])
        count_by_lang[lang] += 1
        emotion_counts[row["emotion"]] += 1
        style_counts[row["style"]] += 1
        licenses_by_lang[lang].add(row["license"])
        sources_by_lang[lang].add(row["source_url"])

    metadata_path = train_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    stats = {
        "duration_by_lang": dict(duration_by_lang),
        "count_by_lang": dict(count_by_lang),
        "emotion_counts": dict(emotion_counts),
        "style_counts": dict(style_counts),
        "licenses_by_lang": {k: sorted(v) for k, v in licenses_by_lang.items()},
        "n_sources_by_lang": {k: len(v) for k, v in sources_by_lang.items()},
    }
    return metadata_path, stats


def write_dataset_card(stats: dict):
    lines = []
    lines.append("---")
    lines.append("license: cc-by-4.0")
    lines.append("language:")
    for lang in config.LANGUAGES:
        lines.append(f"  - {lang.split('-')[0]}")
    lines.append("tags:")
    lines.append("  - text-to-speech")
    lines.append("  - tts")
    lines.append("  - indian-languages")
    lines.append("  - emotion")
    lines.append("task_categories:")
    lines.append("  - text-to-speech")
    lines.append("---\n")

    lines.append("# Indian Multilingual Emotion-Tagged TTS Dataset\n")
    lines.append(
        "Single-speaker speech clips in Indian English and "
        f"{', '.join(v['name'] for k, v in config.LANGUAGES.items() if k != 'en-IN')}, "
        "with transcripts and emotion/style tags, built for TTS training.\n"
    )

    lines.append("## Dataset summary\n")
    total_minutes = sum(stats["duration_by_lang"].values()) / 60
    lines.append(f"- **Total duration:** {total_minutes:.1f} minutes")
    for lang, secs in stats["duration_by_lang"].items():
        name = config.LANGUAGES.get(lang, {}).get("name", lang)
        n = stats["count_by_lang"][lang]
        lines.append(f"  - {name} ({lang}): {secs/60:.1f} min, {n} clips")
    lines.append("")

    lines.append("## Fields\n")
    lines.append("| field | description |")
    lines.append("|---|---|")
    lines.append("| `file_name` | relative path to the audio clip (AudioFolder convention) |")
    lines.append("| `text` | transcript, produced via Sarvam AI ASR with diarization |")
    lines.append("| `language` | Sarvam language code, e.g. `en-IN`, `bn-IN` |")
    lines.append("| `emotion` | one of: " + ", ".join(config.EMOTION_TAGS) + " |")
    lines.append("| `style` | one of: " + ", ".join(config.STYLE_TAGS) + " |")
    lines.append("| `speaker_id` | anonymized per-source-per-speaker identifier (not a real name) |")
    lines.append("| `duration_sec` | clip duration in seconds |")
    lines.append("| `source_url` | original source URL, for attribution |")
    lines.append("| `license` | license basis for this clip: " + ", ".join(config.ALLOWED_LICENSES) + " |")
    lines.append("")

    lines.append("## Construction\n")
    lines.append(
        "Audio was sourced from YouTube videos under Creative Commons / explicit "
        "creator permission, plus self-recorded material, then:\n"
    )
    lines.append("1. Loudness-normalized (EBU R128) and resampled")
    lines.append("2. Transcribed and diarized with Sarvam AI's Saaras v3 batch STT API")
    lines.append("3. Split into single-speaker clips by merging consecutive same-speaker turns")
    lines.append(
        f"4. Auto-filtered for audio quality (SNR >= {config.MIN_SNR_DB} dB, "
        f"clipping ratio <= {config.MAX_CLIPPING_RATIO}, "
        f"duration {config.MIN_CLIP_SECONDS}-{config.MAX_CLIP_SECONDS}s)"
    )
    lines.append("5. Emotion/style auto-tagged via Sarvam chat completions over the transcript text,")
    lines.append("   with a manually-reviewed sample for correction")
    lines.append("")

    lines.append("## Known limitations\n")
    lines.append("- Emotion/style tags are inferred from transcript text by an LLM, not labeled from")
    lines.append("  the audio directly; delivery that contradicts the words (sarcasm, deadpan) may be")
    lines.append("  mis-tagged. A manually-reviewed sample is included; full manual verification of every")
    lines.append("  clip was not performed.")
    lines.append("- Speaker identity is not verified beyond diarization; treat `speaker_id` as a grouping")
    lines.append("  key, not a verified real-world identity.")
    lines.append("- Background noise/quality varies by source even after automated filtering.")
    lines.append("")

    lines.append("## Licensing\n")
    for lang, licenses in stats["licenses_by_lang"].items():
        lines.append(f"- {lang}: {', '.join(licenses)}")
    lines.append(
        "\nEach clip's `license` field traces its legal basis. See the project's "
        "`sources.csv` and `proof/` for attribution and permission records."
    )

    (config.FINAL_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def push_to_hub(repo_id: str):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(config.FINAL_DIR),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo", default=None, help="e.g. your-username/indian-tts-dataset")
    args = parser.parse_args()

    tagged_csv = config.CLIPS_DIR / "clip_manifest_tagged.csv"
    if not tagged_csv.exists():
        print("Run src/emotion_tag.py first.")
        return

    metadata_path, stats = build_local_dataset(tagged_csv)
    write_dataset_card(stats)

    print(f"\nLocal dataset assembled at {config.FINAL_DIR}")
    print(f"Metadata: {metadata_path}")
    print("\nStats:")
    for lang, secs in stats["duration_by_lang"].items():
        print(f"  {lang}: {secs/60:.1f} min, {stats['count_by_lang'][lang]} clips, "
              f"{stats['n_sources_by_lang'][lang]} source(s)")

    if args.push:
        if not args.repo:
            print("\n--push requires --repo your-username/dataset-name")
            return
        push_to_hub(args.repo)
    else:
        print("\nReview data/final/ and the README.md, then re-run with:")
        print("  python -m src.build_hf_dataset --push --repo your-username/indian-tts-dataset")


if __name__ == "__main__":
    main()
