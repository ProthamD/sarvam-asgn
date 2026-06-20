"""
pipeline/emotion_tagger.py — Emotion and speaking style tagging via Sarvam LLM.

Uses the Sarvam chat completions API (OpenAI-compatible) to classify each
segment's transcript into:
  - primary_emotion: neutral | happy | sad | excited | angry | fearful | surprised | calm
  - speaking_style: formal | informal | narrative | conversational | instructional |
                    interview | debate | prayer | news_reading | storytelling | whisper
  - speech_rate:    slow | normal | fast

Segments are batched (10 per LLM call) to minimize API costs.
Uses structured JSON output with few-shot examples for consistency.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from rich.progress import track

from pipeline.utils import (
    get_logger, load_config, get_path, get_env,
    save_json, load_json, append_jsonl
)

logger = get_logger("emotion_tagger")

# ── Valid tag values ──────────────────────────────────────────────────────────
VALID_EMOTIONS = {
    "neutral", "happy", "sad", "excited", "angry",
    "fearful", "surprised", "calm", "serious"
}
VALID_STYLES = {
    "formal", "informal", "narrative", "conversational", "instructional",
    "interview", "debate", "prayer", "news_reading", "storytelling", "whisper"
}
VALID_RATES = {"slow", "normal", "fast"}

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert audio dataset annotator specializing in Indian speech data.
Your task is to analyze speech transcriptions and assign accurate emotion, style, and pace tags
for a Text-to-Speech (TTS) training dataset.

OUTPUT FORMAT — always respond with a RAW JSON array ONLY, one object per input.
DO NOT OUTPUT ANY OTHER TEXT. NO REASONING. NO MARKDOWN CODE BLOCKS.
[
  {
    "segment_id": "...",
    "primary_emotion": "<one of: neutral|happy|sad|excited|angry|fearful|surprised|calm|serious>",
    "speaking_style": "<one of: formal|informal|narrative|conversational|instructional|interview|debate|prayer|news_reading|storytelling|whisper>",
    "speech_rate": "<one of: slow|normal|fast>",
    "confidence": <0.0-1.0>
  }
]
"""


# ── LLM client ────────────────────────────────────────────────────────────────

def _get_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://api.sarvam.ai/v1",
    )


def _default_tag(segment_id: str) -> dict:
    return {
        "segment_id": segment_id,
        "primary_emotion": "neutral",
        "speaking_style": "conversational",
        "speech_rate": "normal",
        "confidence": 0.1,
        "reasoning": "default (LLM call failed or empty transcript)",
    }


def _validate_tag(tag: dict, segment_id: str) -> dict:
    """Ensure all fields are valid; fix or default invalid values."""
    tag["segment_id"] = segment_id
    tag["primary_emotion"] = tag.get("primary_emotion", "neutral")
    if tag["primary_emotion"] not in VALID_EMOTIONS:
        tag["primary_emotion"] = "neutral"
    tag["speaking_style"] = tag.get("speaking_style", "conversational")
    if tag["speaking_style"] not in VALID_STYLES:
        tag["speaking_style"] = "conversational"
    tag["speech_rate"] = tag.get("speech_rate", "normal")
    if tag["speech_rate"] not in VALID_RATES:
        tag["speech_rate"] = "normal"
    tag["confidence"] = float(tag.get("confidence", 0.5))
    tag["confidence"] = max(0.0, min(1.0, tag["confidence"]))
    tag["reasoning"] = tag.get("reasoning", "")[:200]
    return tag


def _tag_batch(
    client: OpenAI,
    model: str,
    batch: list[dict],  # list of {segment_id, transcript, language_code, duration_s}
) -> list[dict]:
    """Send a batch of up to 10 segments to the LLM for tagging."""
    # Build user message
    items = []
    for i, seg in enumerate(batch):
        lang_hint = "(Hindi)" if "hi" in seg.get("language_code", "") else "(Indian English)"
        items.append(
            f'[{i+1}] segment_id="{seg["segment_id"]}" {lang_hint}\n'
            f'Transcript: "{seg.get("transcript", "")}"'
        )

    user_msg = "Analyze these speech segments and return the JSON array:\n\n" + "\n\n".join(items)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        msg = response.choices[0].message
        # sarvam-30b is a reasoning model: JSON may be in content or reasoning_content
        content = msg.content
        if not content and hasattr(msg, 'reasoning_content'):
            content = msg.reasoning_content or ""
        if not content:
            raise ValueError("Empty response from LLM")
        content = content.strip()

        # Extract JSON array from response
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON array found in response: {content[:200]}")

        tags_raw = json.loads(json_match.group())
        # Map segment_id → tag
        id_to_tag: dict[str, dict] = {}
        for tag in tags_raw:
            if isinstance(tag, dict):
                sid = tag.get("segment_id", "")
                id_to_tag[sid] = tag

        # Build result list, using defaults for missing
        result = []
        for seg in batch:
            sid = seg["segment_id"]
            tag = id_to_tag.get(sid) or {}
            # Try positional fallback if segment_id not matched
            if not tag and len(tags_raw) == len(batch):
                tag = tags_raw[batch.index(seg)]
            result.append(_validate_tag(tag, sid))
        return result

    except Exception as e:
        logger.warning(f"LLM batch tagging failed: {e}")
        return [_default_tag(seg["segment_id"]) for seg in batch]


# ── Main tagging function ─────────────────────────────────────────────────────

def tag_all(
    segments: list[dict],
    config: dict | None = None,
    batch_size: int = 4,
    force: bool = False,
) -> list[dict]:
    """
    Add emotion/style tags to all segments.
    Returns segments with tag fields added.
    """
    cfg = config or load_config()
    meta_dir = get_path("metadata", cfg)
    meta_dir.mkdir(parents=True, exist_ok=True)

    tags_path = meta_dir / "emotion_tags.jsonl"
    done_manifest = meta_dir / "tagged_segments.json"

    # Load cached tags
    done_ids: set[str] = set()
    existing_tags: dict[str, dict] = {}
    if not force and done_manifest.exists():
        done_data = load_json(done_manifest)
        for item in done_data:
            sid = item["segment_id"]
            done_ids.add(sid)
            existing_tags[sid] = item
        logger.info(f"[skip] {len(done_ids)} segments already tagged")

    api_key = get_env("SARVAM_API_KEY")
    model = cfg["sarvam"]["llm_model"]
    client = _get_client(api_key)

    # Segments that need tagging
    to_tag = [s for s in segments if s["segment_id"] not in done_ids or force]
    logger.info(f"[tag] Tagging {len(to_tag)} segments with {model} (batch_size={batch_size})")

    all_new_tags: list[dict] = []

    for i in range(0, len(to_tag), batch_size):
        batch = to_tag[i: i + batch_size]
        batch_input = [
            {
                "segment_id": s["segment_id"],
                "transcript": s.get("transcript", ""),
                "language_code": s.get("language_code", "en-IN"),
                "duration_s": s.get("duration_s", 0),
            }
            for s in batch
        ]
        tags = _tag_batch(client, model, batch_input)
        all_new_tags.extend(tags)

        # Small rate-limit delay
        if i + batch_size < len(to_tag):
            time.sleep(0.5)

    # Merge tags into segments
    new_tag_map: dict[str, dict] = {t["segment_id"]: t for t in all_new_tags}

    tagged_segments: list[dict] = []
    for seg in segments:
        sid = seg["segment_id"]
        tag = new_tag_map.get(sid) or existing_tags.get(sid) or _default_tag(sid)
        seg["primary_emotion"] = tag["primary_emotion"]
        seg["speaking_style"] = tag["speaking_style"]
        seg["speech_rate"] = tag["speech_rate"]
        seg["tag_confidence"] = tag["confidence"]
        seg["tag_reasoning"] = tag.get("reasoning", "")
        tagged_segments.append(seg)

    # Save
    save_json(tagged_segments, done_manifest)

    if tags_path.exists():
        tags_path.unlink()
    for seg in tagged_segments:
        append_jsonl({
            "segment_id": seg["segment_id"],
            "transcript": seg.get("transcript", ""),
            "language_code": seg.get("language_code", ""),
            "primary_emotion": seg.get("primary_emotion", "neutral"),
            "speaking_style": seg.get("speaking_style", "conversational"),
            "speech_rate": seg.get("speech_rate", "normal"),
            "tag_confidence": seg.get("tag_confidence", 0.0),
            "tag_reasoning": seg.get("tag_reasoning", ""),
        }, tags_path)

    # Print distribution
    emotion_counts: dict[str, int] = {}
    style_counts: dict[str, int] = {}
    for seg in tagged_segments:
        emotion_counts[seg.get("primary_emotion", "?")] = emotion_counts.get(seg.get("primary_emotion", "?"), 0) + 1
        style_counts[seg.get("speaking_style", "?")] = style_counts.get(seg.get("speaking_style", "?"), 0) + 1

    logger.info("[emotion distribution]")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {emotion:<20} {count}")
    logger.info("[style distribution]")
    for style, count in sorted(style_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {style:<20} {count}")

    return tagged_segments
