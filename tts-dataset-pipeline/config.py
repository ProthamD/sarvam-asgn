"""
Central configuration for the Indian-language TTS dataset pipeline.

Edit LANGUAGES below to change which two languages you build. Everything
else (paths, models, taxonomy) is shared across languages.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Languages to build. Key = Sarvam language_code, value = human-readable name
# and the BCP-47-ish tag used in the published HF dataset.
#
# Bengali is set as the default second language since it pairs naturally with
# Indian English for this dataset, but any Sarvam-supported code works:
# hi-IN, ta-IN, te-IN, kn-IN, ml-IN, mr-IN, gu-IN, pa-IN, or-IN, bn-IN ...
# ---------------------------------------------------------------------------
LANGUAGES = {
    "en-IN": {"name": "Indian English", "hf_tag": "en-IN", "target_minutes": 30},
    "bn-IN": {"name": "Bengali", "hf_tag": "bn-IN", "target_minutes": 30},
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # full downloaded audio, per source
CHUNKS_DIR = DATA_DIR / "chunks"           # raw audio split into <=50min pieces
ASR_DIR = DATA_DIR / "asr_raw"             # raw Sarvam diarized STT JSON
CLIPS_DIR = DATA_DIR / "clips"             # final single-speaker clip wavs
REVIEW_DIR = DATA_DIR / "review"           # low-confidence clips for manual QA
FINAL_DIR = DATA_DIR / "final"             # HF AudioFolder-ready dataset
LOG_DIR = ROOT / "logs"

for d in [RAW_DIR, CHUNKS_DIR, ASR_DIR, CLIPS_DIR, REVIEW_DIR, FINAL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sarvam API
# ---------------------------------------------------------------------------
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_STT_MODEL = "saaras:v3"          # current recommended ASR model
SARVAM_CHAT_MODEL = "sarvam-30b"        # use sarvam-105b for higher quality / cost
SARVAM_BATCH_MAX_SECONDS = 50 * 60      # keep margin under the documented 1hr cap

# ---------------------------------------------------------------------------
# Audio targets
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 22050              # good default for TTS training (Sarvam ASR is fine down/upsampling to 16k internally)
ASR_SAMPLE_RATE = 16000                 # Sarvam STT works best at 16kHz
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 20.0                 # merge/split turns to land in this band
TARGET_CLIP_SECONDS = 8.0               # soft target when merging consecutive turns
SILENCE_PAD_MS = 150

# Quality gates (see src/quality_filter.py)
MIN_SNR_DB = 15.0
MAX_CLIPPING_RATIO = 0.001              # fraction of samples allowed at +/-1.0
MIN_ASR_CONFIDENCE = 0.0                # Saaras v3 does not always return confidence; kept as a hook

# ---------------------------------------------------------------------------
# Emotion / style taxonomy
# Keep this closed-set so tags stay consistent across the whole dataset.
# ---------------------------------------------------------------------------
EMOTION_TAGS = [
    "neutral", "happy", "sad", "angry", "excited", "fearful",
    "surprised", "disgusted", "calm",
]
STYLE_TAGS = [
    "conversational", "formal", "narrative", "instructional",
    "whisper", "shouting", "questioning", "emphatic", "sarcastic",
]

# ---------------------------------------------------------------------------
# Licensing — every source MUST have one of these recorded per clip.
# Do not add a source that does not clearly fall into one of these buckets.
# ---------------------------------------------------------------------------
ALLOWED_LICENSES = [
    "cc-by",            # YouTube "Creative Commons" filter, attribution required
    "cc0-public-domain",
    "self-recorded",    # you or a consenting collaborator recorded it
    "explicit-permission",  # written permission from the rights holder, keep proof
]
