# Manual QA Checklist (do this before publishing)

Automated checks (`quality_filter.py`, `emotion_tag.py`) catch the obvious
problems but won't catch everything. Before you push to the Hub, spend
30-45 minutes on this.

## 1. Listen to a random sample
Pick ~20 clips per language at random (not just the first N — shuffle).
For each, check:
- [ ] Transcript matches what's actually said (Sarvam ASR isn't perfect,
      especially on code-mixed or accented speech)
- [ ] Only one speaker audible, no crosstalk or bleed from a second voice
- [ ] No abrupt cut mid-word at clip start/end (if so, widen
      `SILENCE_PAD_MS` in config.py and re-cut, or hand-trim)
- [ ] No background music/noise that slipped past the SNR filter

## 2. Review the emotion/style tag sample
Open `data/clips/tag_review_sample.csv`, listen to each clip, and correct
the `emotion`/`style` columns if the auto-tag (text-only) doesn't match
the actual delivery. Then manually propagate any systematic corrections
back into `clip_manifest_tagged.csv` before running `build_hf_dataset.py`
(e.g. if everything from one self-recorded "angry" take got tagged
"neutral" because the words themselves were calm, fix that whole batch).

## 3. Check tag balance
`emotion_tag.py` prints a distribution at the end. If one tag (usually
"neutral") is >60% of the dataset, your self-recorded material isn't
covering the other emotions enough — go back and record a few more
deliberately angry/excited/whisper/etc. takes per language.

## 4. Check duration balance
Each language should land close to its `target_minutes` in config.py.
Check `build_hf_dataset.py`'s printed stats. If one language is short,
add more sources for that language and re-run the pipeline from
`download.py` for just the new rows.

## 5. Verify licensing one more time
For every distinct `source_url` in the final manifest, re-open the
original video and re-confirm the license is still what you recorded
in `sources.csv` (creators occasionally change a video's license after
upload). For `explicit-permission` sources, confirm the proof file
still exists under `proof/`.

## 6. Spot-check the published HF dataset
After `build_hf_dataset.py --push`, open the dataset on huggingface.co,
play 3-4 random clips through the dataset viewer, and confirm the
metadata.csv columns render correctly (file_name resolves to playable
audio, text/emotion/style columns show up).
