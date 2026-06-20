# Indian TTS Dataset Pipeline: Project Report

## Overview
This project presents an automated, production-ready pipeline for generating a high-quality Text-to-Speech (TTS) dataset from YouTube videos. The pipeline supports Indian languages (Indian English and Hindi) and includes advanced processing such as speaker diarization, smart segmentation, acoustic quality filtering, transcription, and emotion tagging.

## Important Links
- **Hugging Face Dataset:** [ProthamD/indian-tts-60min](https://huggingface.co/datasets/ProthamD/indian-tts-60min)
- **GitHub Repository:** [ProthamD/sarvam-asgn](https://github.com/ProthamD/sarvam-asgn)

## Dataset Metrics
The final dataset exceeds the 60-minute target duration with the following metrics:
- **Total Duration:** 1 hour, 18 minutes, 23 seconds
- **Total Segments:** 267 segments
- **Train Split:** 240 segments (1 hr 10 mins)
- **Validation Split:** 27 segments (8 mins)
- **Human Review:** A subset of 6 segments underwent human-in-the-loop quality review and editing. The rest were processed fully automatically.

## Pipeline Architecture
The pipeline consists of 7 sequential stages:
1. **Download:** Audio extraction from YouTube using `yt-dlp` (16kHz mono WAV).
2. **Diarize:** Speaker identification and timestamping using Sarvam Batch API (saaras:v3).
3. **Segment:** Smart merging and splitting of audio chunks, with loudness normalization (EBU R128, -23 LUFS).
4. **Quality Filter:** Automated gating based on acoustic metrics (SNR > 18 dB, spectral flatness < 0.18 to reject music, clipping < 0.2%).
5. **Transcribe:** Text transcription utilizing Sarvam ASR APIs.
6. **Emotion Tagging:** LLM-based tagging (sarvam-m) to categorize primary emotion, speaking style, and speech rate.
7. **Build & Push:** Automated train/validation splitting and direct upload to the Hugging Face Hub.

## Tech Stack
- **APIs:** Sarvam AI (ASR, Diarization, LLM), Hugging Face Hub
- **Libraries:** Python, FFmpeg, yt-dlp, pydub, librosa, Hugging Face `datasets`
