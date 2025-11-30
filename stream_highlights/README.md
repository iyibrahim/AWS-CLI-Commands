# Stream Highlight Automation (Local)

This guide describes a local-only workflow for turning an 8-hour daily stream into compelling stories (up to 20 minutes each). It pairs deterministic media tools (FFmpeg) with lightweight, locally run AI models for transcription and summarization so the workflow can run entirely on your laptop.

## Overview
1. **Ingest**: Pull the raw stream recording (e.g., OBS output file) into a working directory.
2. **Segment**: Cut the source into <=20-minute chunks (aligned to scene/volume boundaries where possible).
3. **Transcribe**: Use a local Whisper model to obtain word-level timestamps and speaker segments.
4. **Score & Select**: Rank segments using heuristics (e.g., chat spikes, volume bursts) and keep the top stories.
5. **Summarize**: Generate narrative summaries and thumbnails locally for the selected chunks.
6. **Render**: Re-encode the selected clips and package them with subtitles and metadata.

The companion `pipeline.py` script wires these stages together so you can run `python pipeline.py --input path/to/stream.mp4` and receive summarized clips in `output/`.

## Prerequisites
- Python 3.10+
- FFmpeg installed and on `$PATH`.
- Local models (suggestions):
  - Whisper: `small` or `medium` (CPU) or `medium.en`/`large-v3` (GPU) via [openai/whisper](https://github.com/openai/whisper) or [faster-whisper](https://github.com/guillaumekln/faster-whisper).
  - Summarization: BART (`facebook/bart-large-cnn`) or LLaMA-3/4-bit via [transformers](https://github.com/huggingface/transformers) or [llama.cpp](https://github.com/ggerganov/llama.cpp).
- Optional: `numpy` (for volume-based scoring), `pysrt` (subtitle post-processing).

## Quickstart (do-this-first)
You can follow these steps without writing code. Start a terminal where your 8-hour recording (e.g., `mystream.mp4`) is stored.

1) **Set up once**
- Create a virtual environment and install the required packages:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install ffmpeg-python numpy torch transformers openai-whisper pysrt
  ```

2) **Run the pipeline on your recording**
- No copying or code editing required—just run the command from the same folder where `stream_highlights/pipeline.py` lives (th
is repo root).
- Replace `/path/to/mystream.mp4` with the actual path to your 8-hour file. Examples:
  - If the file sits next to the repo: `--input /home/you/Videos/mystream.mp4`
  - If the file is in the repo folder: `--input ./mystream.mp4`
- Then run (single line avoids copy/paste issues):
  ```bash
  python3 stream_highlights/pipeline.py --input /path/to/mystream.mp4 --output ./output --max-seconds 1200 --top-k 4 --model-size medium
  ```

Troubleshooting common shell errors:
- `Permission denied`: ensure you are **prefixing the command with `python3`** (or run `chmod +x stream_highlights/pipeline.py` and execute `./stream_highlights/pipeline.py ...`).
- `cannot execute binary file`: the shell tried to run your video as a program—double-check that every line continues with a backslash (`\`) or use the single-line command above.
- `Model ... not found`: remove any accidental trailing characters (e.g., `medium~` ➜ `medium`) and choose a supported Whisper size: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`, `large`, `large-v3-turbo`, `turbo`.

3) **Collect your highlights**
- When it finishes, grab the clips from `output/clips/` and the summaries/metadata from `output/stories.jsonl`.

What the script does for you:
- Creates 20-minute-or-shorter clip candidates aligned to volume dips.
- Transcribes each candidate and saves `.json` transcripts.
- Scores candidates and keeps the top `--top-k` clips.
- Summarizes each kept clip into a story outline and thumbnail prompt.

Time estimates (approximate, depends on your laptop):
- CPU only: ~1.5–2.5× real time (8 hr stream ≈ 12–20 hrs). Use the `small` or `medium` Whisper model to finish sooner.
- GPU (NVIDIA): often <0.6× real time (8 hr stream ≈ 5 hrs or less) with `medium`/`medium.en`.

## Workflow Details
### 1) Segmentation
- Default segmentation uses volume energy to avoid cutting mid-sentence: the script computes frame-level RMS, finds low-energy valleys, and trims to <= `--max-seconds` windows.
- Override with strict fixed windows via `--fixed-slices`.

### 2) Transcription
- Uses Whisper with word-level timestamps. Change model size via `--model-size`.
- If you prefer `faster-whisper`, swap the import inside `transcribe_chunk` and set `--model-size medium.en`.

### 3) Scoring
- Combines three signals (all local):
  - **Pacing**: speaking rate spikes (words/min).
  - **Volume**: RMS peaks (emotional/exciting moments).
  - **Continuity**: penalize segments with long silences.
- Scores are normalized to `[0,1]` so you can extend with chat/overlay metrics if available.

### 4) Summarization & Metadata
- Summaries are generated locally (default: BART). Outputs include:
  - `summary`: 1–3 sentence hook.
  - `beats`: bullet list of the narrative beats.
  - `thumbnail_prompt`: short prompt for a thumbnail generator.
- All metadata is stored in `output/stories.jsonl` for reuse.

### 5) Rendering
- Clips are re-encoded with embedded subtitles using FFmpeg; you can change the codec flags in `render_story_clip`.

## Operational Tips
- Run on GPU if available (`torch.cuda.is_available()`) to keep under real-time.
- Keep a rolling cache of models (avoid re-downloads) in `~/.cache/huggingface` and `~/.cache/whisper`.
- For daily automation, pair with `cron` or a local scheduler (e.g., `systemd` user timer) to kick off the job after each stream ends.
- Monitor disk space; 8 hours of 1080p60 can exceed 40GB before clipping.

## Outputs
- `output/clips/*.mp4`: top stories (<=20 mins) with burned-in subtitles.
- `output/transcripts/*.json`: word-level transcripts per candidate.
- `output/stories.jsonl`: per-story metadata (score, summary, beats, thumbnail prompt, source range).

## Extending
- **Scene changes**: replace volume-based cut detection with `ffprobe` scene change detection (`-vf select='gt(scene,0.4)',showinfo`).
- **Speaker diarization**: integrate `pyannote.audio` locally for multi-speaker labeling.
- **A/B scoring**: keep top-N clips per heuristic (volume vs. pacing) and combine via reciprocal rank fusion.
- **UI**: expose `stories.jsonl` to a minimal React/Streamlit UI for manual approval before rendering.
