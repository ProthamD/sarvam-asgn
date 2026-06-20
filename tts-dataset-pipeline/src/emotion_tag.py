"""
src/emotion_tag.py

Reads clip_manifest_passed.csv and asks Sarvam's chat-completions model to
assign one emotion tag and one style tag per clip, from the closed taxonomy
in config.py (EMOTION_TAGS / STYLE_TAGS). The model only sees the
transcript text (not the audio) -- this is a reasonable proxy for textual
affect, but it WILL miss cases where the delivery contradicts the words
(e.g. sarcasm, deadpan). Tags are written with a "tag_source" column so
you know what's auto vs hand-corrected, and a sample is always pulled out
for manual review -- see the print-out at the end and QUALITY_CHECKLIST.md.

Usage:
    export SARVAM_API_KEY=sk_xxx
    python -m src.emotion_tag
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

SYSTEM_PROMPT = """You are an annotator labeling short transcripts for a TTS \
emotion/style dataset. Given one transcript, pick exactly ONE emotion tag and \
exactly ONE style tag from these closed sets:

EMOTION (pick one): {emotions}
STYLE (pick one): {styles}

Base your judgment only on the wording, punctuation, and phrasing of the \
transcript (you cannot hear the audio). If nothing stands out, use \
"neutral" for emotion and "conversational" for style.

Respond with ONLY a JSON object, no markdown fences, no extra text:
{{"emotion": "<one of the emotion tags>", "style": "<one of the style tags>"}}
"""


def get_client():
    from sarvamai import SarvamAI
    return SarvamAI(api_subscription_key=config.SARVAM_API_KEY)


def tag_one(client, text: str, retries: int = 3) -> dict:
    system_prompt = SYSTEM_PROMPT.format(
        emotions=", ".join(config.EMOTION_TAGS),
        styles=", ".join(config.STYLE_TAGS),
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                model=config.SARVAM_CHAT_MODEL,
                temperature=0.2,
                max_tokens=60,
            )
            content = response.choices[0].message.content.strip()
            content = content.strip("`").removeprefix("json").strip()
            parsed = json.loads(content)
            emotion = parsed.get("emotion", "neutral")
            style = parsed.get("style", "conversational")
            if emotion not in config.EMOTION_TAGS:
                emotion = "neutral"
            if style not in config.STYLE_TAGS:
                style = "conversational"
            return {"emotion": emotion, "style": style}
        except Exception as e:
            print(f"[retry {attempt+1}/{retries}] tagging failed: {e}")
            time.sleep(2)
    return {"emotion": "neutral", "style": "conversational"}


def main():
    passed_path = config.CLIPS_DIR / "clip_manifest_passed.csv"
    if not passed_path.exists():
        print("Run src/quality_filter.py first.")
        return
    if not config.SARVAM_API_KEY:
        print("Set SARVAM_API_KEY in your environment first.")
        return

    client = get_client()
    tagged_path = config.CLIPS_DIR / "clip_manifest_tagged.csv"

    with open(passed_path, newline="", encoding="utf-8") as in_f:
        rows = list(csv.DictReader(in_f))

    fieldnames = list(rows[0].keys()) + ["emotion", "style", "tag_source"]
    with open(tagged_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            text = row["text"].strip()
            if not text:
                row["emotion"], row["style"], row["tag_source"] = "neutral", "conversational", "auto-empty"
            else:
                tags = tag_one(client, text)
                row["emotion"], row["style"], row["tag_source"] = tags["emotion"], tags["style"], "auto"
            writer.writerow(row)
            if i % 25 == 0:
                print(f"...tagged {i}/{len(rows)}")

    # pull a manual-review sample: a few per emotion tag, capped
    sample_rows = []
    by_emotion = {}
    with open(tagged_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_emotion.setdefault(row["emotion"], []).append(row)
    for emo, group in by_emotion.items():
        sample_rows.extend(random.sample(group, min(3, len(group))))

    review_sample_path = config.CLIPS_DIR / "tag_review_sample.csv"
    if sample_rows:
        with open(review_sample_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sample_rows[0].keys())
            writer.writeheader()
            writer.writerows(sample_rows)

    print(f"\nDone. Tagged manifest: {tagged_path}")
    print(f"Manual-review sample (listen + correct these by hand): {review_sample_path}")
    print("\nTag distribution:")
    for emo, group in sorted(by_emotion.items(), key=lambda x: -len(x[1])):
        print(f"  {emo}: {len(group)}")
    print("\nNext: hand-correct tag_review_sample.csv as needed, then python -m src.build_hf_dataset")


if __name__ == "__main__":
    main()
