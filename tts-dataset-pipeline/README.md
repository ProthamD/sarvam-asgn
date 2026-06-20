# Indian-Language TTS Dataset Pipeline

Builds a ~60-minute, single-speaker, emotion/style-tagged TTS dataset
(30 min Indian English + 30 min a second Indian language, default Bengali)
sourced from YouTube + self-recordings, transcribed/diarized via Sarvam AI,
and published as a public Hugging Face dataset.

## Setup

```bash
cd tts-dataset-pipeline
pip install -r requirements.txt --break-system-packages
# also needs ffmpeg + yt-dlp's runtime deps on the system (ffmpeg is usually preinstalled)

export SARVAM_API_KEY=sk_xxx        # from dashboard.sarvam.ai
export HF_TOKEN=hf_xxx              # only needed for the final --push step
```

## Run order

1. **Read `SOURCING_GUIDE.md`**, then fill in `sources.csv` with real,
   license-clean video URLs (or plan your self-recordings).
2. `python -m src.download` - pulls audio, skips any row without a
   verified license
3. `python -m src.preprocess` - loudness-normalizes, chunks to fit
   Sarvam's batch size limit
4. `python -m src.transcribe_diarize` - Sarvam STT + diarization +
   timestamps per chunk
5. `python -m src.segment_by_speaker` - cuts single-speaker clips,
   merges short consecutive turns toward ~8s
6. `python -m src.quality_filter` - auto SNR/clipping/duration gate
7. `python -m src.emotion_tag` - LLM-based emotion/style tagging via
   Sarvam chat completions
8. **Do `QUALITY_CHECKLIST.md` by hand** - this step matters, don't skip it
9. `python -m src.build_hf_dataset` - assembles local AudioFolder dataset
   + dataset card
10. `python -m src.build_hf_dataset --push --repo your-username/indian-tts-dataset`
    - publishes to the Hub

## Directory layout produced

```
data/
  raw/          full downloaded source audio + raw_manifest.csv
  chunks/       loudness-normalized, batch-sized audio + chunk_manifest.csv
  asr_raw/      Sarvam diarized STT JSON per chunk + asr_manifest.csv
  clips/        final single-speaker clips + manifests at each filter stage
  review/       chunks Sarvam couldn't diarize, for manual handling
  final/        the published HF AudioFolder dataset (train/<lang>/*.wav, metadata.csv, README.md)
```

## Design notes / why it's built this way

- **Licensing is enforced at the download step, not after.** `download.py`
  refuses to pull a row from `sources.csv` unless it has both a valid
  `license` value and a `license_proof`. This is the single most important
  thing standing between you and a delisted dataset later.
- **Diarization-first, not VAD-first.** Rather than naive silence-based
  chunking, turns come from Sarvam's diarization output so multi-speaker
  source videos (interviews, podcasts) can still be safely mined for their
  single-speaker stretches without manual editing.
- **Quality filter is automated but not authoritative** - see
  `QUALITY_CHECKLIST.md`. SNR/clipping catch the worst clips quickly, but
  a "neutral"-biased emotion distribution is a known failure mode in
  found-audio pipelines, which is why self-recording for rare emotions is
  recommended in `SOURCING_GUIDE.md`.
- **`config.py` is the only place you should need to edit** to swap the
  second language, retarget durations, or change the SNR/clip-length
  thresholds.
