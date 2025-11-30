"""Local stream highlight pipeline.

Usage:
    python pipeline.py --input /path/to/stream.mp4 --output ./output

The pipeline is designed to run fully offline: it leans on FFmpeg for
media handling and local models for ASR and summarization. Heavy imports
are intentionally placed inside functions so the script can be inspected
without requiring optional dependencies until a stage is executed.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Segment:
    """Represents a candidate story segment."""

    start: float
    end: float
    score: float = 0.0
    transcript_path: Path | None = None
    summary: str | None = None
    beats: List[str] | None = None
    thumbnail_prompt: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


# ------------------------- Media utilities -------------------------

def run_ffmpeg(args: Sequence[str]) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-y", *args]
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_rms_peaks(video_path: Path, hop: float = 1.0) -> List[Tuple[float, float]]:
    """Return list of (time, rms) values at 1-second intervals using FFmpeg."""

    try:
        result = run_ffmpeg(
            [
                "-i",
                str(video_path),
                "-vn",
                "-af",
                f"astats=metadata=1:reset={hop}",
                "-f",
                "null",
                "-",
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("FFmpeg astats failed") from exc

    # The astats filter prints to stderr; parse RMS_level lines.
    rms: List[Tuple[float, float]] = []
    log = (result.stderr or b"").decode()
    time_cursor = 0.0
    for line in log.splitlines():
        if "RMS_level" not in line:
            continue
        # RMS_level: -20.0
        try:
            rms_value = float(line.split("RMS_level:")[-1].strip())
            rms.append((time_cursor, rms_value))
            time_cursor += hop
        except ValueError:
            continue
    return rms


def propose_segments(duration: float, max_seconds: int, rms: List[Tuple[float, float]]) -> List[Segment]:
    if not rms:
        return [Segment(0, min(duration, max_seconds))]

    # Find low-energy cut points: times where RMS dips below median.
    _, values = zip(*rms)
    median_rms = sorted(values)[len(values) // 2]

    segments: List[Segment] = []
    window_start = 0.0
    cursor = 0.0
    for t, val in rms:
        cursor = t
        if cursor - window_start >= max_seconds and val <= median_rms:
            segments.append(Segment(window_start, cursor))
            window_start = cursor

    if duration - window_start > 1.0:
        segments.append(Segment(window_start, min(duration, window_start + max_seconds)))

    return segments


def cut_segment(video_path: Path, segment: Segment, output_path: Path) -> None:
    run_ffmpeg(
        [
            "-ss",
            str(segment.start),
            "-i",
            str(video_path),
            "-t",
            str(segment.duration),
            "-c",
            "copy",
            str(output_path),
        ]
    )


# ------------------------- ASR and summarization -------------------------

def transcribe_chunk(audio_path: Path, model_size: str = "medium") -> dict:
    """Return Whisper-style transcription with word-level timestamps."""

    import whisper  # lazy import to keep dependency optional

    model = whisper.load_model(model_size)
    result = model.transcribe(str(audio_path), word_timestamps=True)
    return result


def summarize_text(text: str, max_tokens: int = 220) -> str:
    """Summarize text locally using a transformers pipeline."""

    from transformers import pipeline  # lazy import

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    output = summarizer(text, max_length=max_tokens, min_length=80, do_sample=False)
    return output[0]["summary_text"]


def beats_from_transcript(words: Iterable[str]) -> List[str]:
    chunk = " ".join(words)
    # Lightweight beat extraction: split on sentences.
    beats = [sentence.strip() for sentence in chunk.split(".") if sentence.strip()]
    return beats[:6]


def score_segment(transcript: dict, rms: List[Tuple[float, float]], segment: Segment) -> float:
    words = transcript.get("segments", [])
    total_words = sum(len(seg.get("words", [])) for seg in words)
    duration = max(segment.duration, 1.0)
    words_per_min = total_words / (duration / 60)

    # Volume score: normalize RMS inside this window.
    window_rms = [val for t, val in rms if segment.start <= t <= segment.end]
    if window_rms:
        volume_score = (max(window_rms) - min(window_rms)) if len(window_rms) > 1 else window_rms[0]
    else:
        volume_score = 0.0

    pacing = min(words_per_min / 180, 1.0)  # assume 180 wpm as upper bound
    vol_norm = 1 / (1 + math.exp(-volume_score / 5))  # squashed to [0,1]
    continuity_penalty = 0.0
    for seg in words:
        gaps = [seg.get("words", [])[i + 1]["start"] - seg.get("words", [])[i]["end"] for i in range(len(seg.get("words", [])) - 1)]
        long_gaps = [g for g in gaps if g > 2.5]
        continuity_penalty += 0.05 * len(long_gaps)

    score = max(0.0, min(1.0, 0.4 * pacing + 0.5 * vol_norm - continuity_penalty))
    return score


# ------------------------- Rendering -------------------------

def render_story_clip(video_path: Path, transcript: dict, segment: Segment, output_path: Path) -> None:
    # Dump subtitles to a temporary SRT file.
    srt_lines: List[str] = []
    counter = 1
    for seg in transcript.get("segments", []):
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        srt_lines.append(f"{counter}\n{format_ts(start)} --> {format_ts(end)}\n{text}\n\n")
        counter += 1

    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode="w", encoding="utf-8") as srt_tmp:
        srt_tmp.writelines(srt_lines)
        srt_path = Path(srt_tmp.name)

    run_ffmpeg(
        [
            "-ss",
            str(segment.start),
            "-i",
            str(video_path),
            "-t",
            str(segment.duration),
            "-vf",
            f"subtitles={srt_path.as_posix()}:force_style='Fontsize=18,PrimaryColour=&HFFFFFF&'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            str(output_path),
        ]
    )


def format_ts(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ------------------------- Orchestrator -------------------------

def run_pipeline(
    input_path: Path,
    output_dir: Path,
    max_seconds: int = 1200,
    top_k: int = 4,
    fixed_slices: bool = False,
    model_size: str = "medium",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    transcripts_dir = output_dir / "transcripts"
    clips_dir.mkdir(exist_ok=True)
    transcripts_dir.mkdir(exist_ok=True)

    duration = probe_duration(input_path)

    rms = extract_rms_peaks(input_path)
    if fixed_slices:
        rms = []  # bypass adaptive cuts

    segments = propose_segments(duration, max_seconds, rms)
    if not segments:
        raise RuntimeError("No segments proposed; check input file or FFmpeg install.")

    stories = []
    for idx, segment in enumerate(segments):
        chunk_path = clips_dir / f"candidate_{idx:03d}.mp4"
        cut_segment(input_path, segment, chunk_path)

        # Transcribe
        transcript = transcribe_chunk(chunk_path, model_size=model_size)
        transcript_path = transcripts_dir / f"candidate_{idx:03d}.json"
        transcript_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
        segment.transcript_path = transcript_path

        # Score
        segment.score = score_segment(transcript, rms, segment)

        # Summarize
        text = " ".join(seg.get("text", "") for seg in transcript.get("segments", []))
        segment.summary = summarize_text(text)
        segment.beats = beats_from_transcript(text.split())
        segment.thumbnail_prompt = f"Exciting moment: {segment.summary[:120]}"

        stories.append((segment.score, segment, transcript, chunk_path))

    # Keep top-k stories
    stories.sort(key=lambda item: item[0], reverse=True)
    kept = stories[:top_k]

    metadata_path = output_dir / "stories.jsonl"
    with metadata_path.open("w", encoding="utf-8") as fh:
        for rank, (score, segment, transcript, chunk_path) in enumerate(kept, start=1):
            output_clip = clips_dir / f"story_{rank:02d}.mp4"
            render_story_clip(input_path, transcript, segment, output_clip)

            record = asdict(segment)
            record.update(
                {
                    "score": score,
                    "output_clip": output_clip.as_posix(),
                    "rank": rank,
                    "source": input_path.as_posix(),
                }
            )
            fh.write(json.dumps(record) + "\n")
            print(f"Saved story {rank} -> {output_clip} (score={score:.3f})")


# ------------------------- CLI -------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local stream highlight automation")
    parser.add_argument("--input", type=Path, required=True, help="Path to source video")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--max-seconds", type=int, default=1200, help="Maximum duration per story in seconds")
    parser.add_argument("--top-k", type=int, default=4, help="Number of stories to keep")
    parser.add_argument("--fixed-slices", action="store_true", help="Disable RMS-based adaptive cuts")
    parser.add_argument("--model-size", type=str, default="medium", help="Whisper model size")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        max_seconds=args.max_seconds,
        top_k=args.top_k,
        fixed_slices=args.fixed_slices,
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
