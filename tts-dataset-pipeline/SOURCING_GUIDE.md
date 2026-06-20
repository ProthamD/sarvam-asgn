# Choosing Source Audio (Read Before Downloading Anything)

A YouTube video being "public" does not make its audio free to redistribute inside
a dataset. Re-publishing someone's voice without the right license is the single
biggest way this kind of project gets a dataset delisted from the Hub later, so
treat this step as seriously as the engineering.

## Pick sources that satisfy ONE of these, and record it in `sources.csv`

1. **`cc-by` — YouTube's Creative Commons filter.** On YouTube search, use
   `Filters → Creative Commons`. Re-confirm on the video's own page (Description
   → "Show more" → license line) since the filter is sometimes stale. CC-BY
   requires attribution — keep the channel name and video URL, you'll credit it
   in the dataset card.
2. **`cc0-public-domain`** — official government archives (e.g. All India Radio
   archival uploads, Parliament/Lok Sabha TV proceedings, Doordarshan archival
   clips where explicitly marked public domain), or recordings whose copyright
   has lapsed.
3. **`self-recorded`** — you (or a friend/classmate who consents) record fresh
   audio: read-aloud passages, monologues in different emotional deliveries,
   a mock interview. This is the most reliable way to guarantee both clean
   audio *and* a clean license, and it's the fastest way to hit specific
   emotion/style tags on demand instead of hoping to find them in the wild.
4. **`explicit-permission`** — you message a creator and they reply in writing
   that you may use and redistribute their audio in an open dataset. Save the
   reply (screenshot or forwarded email) under `proof/`.

Avoid: news broadcasts, film/TV clips, music videos, monetized podcasts/interviews
without checking license, dubbed/AI-voiced content, and anything with background
music or a studio jingle under the speech.

## What "good source audio" looks like technically

- Single visible speaker, talking head or voice-over, no overlapping crosstalk
- No background music, laugh track, or audience noise bed
- Decent mic (lav/headset/podcast mic, not phone-speaker-in-a-room echo)
- A range of delivery styles across your chosen videos so the emotion/style
  tags in the final dataset aren't 90% "neutral" — see the suggestions below.

## Suggested content types per language

**Indian English**
- Solo TEDx/TEDx-style or conference talks (many are CC-BY) → formal, narrative, excited
- Personal vlogs / study-with-me / tech explainer channels → conversational, calm, instructional
- Audiobook or poetry reading channels with open licensing → narrative, whisper, emphatic
- A self-recorded set: same paragraph read in 4–5 different emotions to deliberately
  fill underrepresented tags (angry, fearful, surprised, sarcastic are rare in the wild)

**Bengali** (or your chosen language — same logic applies)
- All India Radio Bengali archival talks/natoks where marked public domain
- Bengali solo vlogs / education / motivational-speaking channels with CC-BY or permission
- Self-recorded readings of Bengali short stories/poems across emotions — also the
  most reliable way to get natural, accurate (non-ASR-error) Bengali transcripts,
  since you can write the ground-truth text yourself before recording and only use
  Sarvam ASR as a cross-check, not the sole source of truth.

## A practical mix that gets you to ~30 minutes per language without much
## hunting

- 3–4 found CC-BY/permission videos, ~5–8 minutes of *usable single-speaker*
  audio each (a 20-minute talk rarely yields 20 clean minutes after removing
  audience laughter, Q&A crosstalk, and intro music)
- Plus 8–10 self-recorded minutes specifically targeting the rarer emotion/style
  tags (whisper, shouting, sarcastic, fearful, surprised) — these are
  genuinely hard to source from YouTube and easy to record deliberately.

Document every source in `sources.csv` before running `src/download.py`.
