"""Japanese karaoke video generator pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import difflib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

REPO_DIR = Path(__file__).resolve().parent
CACHE_DIR = REPO_DIR / "cache"

# Redirect library caches before importing torch or downstream packages.
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["HF_HOME"] = str(CACHE_DIR / "hf")

from fontTools.ttLib import TTFont
import fugashi
import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import resample_poly
import soundfile as sf
import stable_whisper
import torch
import torchcrepe

STAGES = ["decode", "separate", "align", "ass", "guide", "bg", "render"]

DEFAULT_SEP_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/Japanese/"
    "NotoSansCJKjp-Bold.otf"
)
DEFAULT_FONT_FILENAME = "NotoSansCJKjp-Bold.otf"

SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
NON_SUNG_CHARS = set(
    " 　\t\n\r!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～"
    "、。・…‥「」『』（）〔〕［］｛｝〈〉《》【】〖〗〘〙〚〛〜～〽゠＝"
)


def parse_volume(val: str) -> float:
    s = val.strip()
    if not s:
        raise argparse.ArgumentTypeError("Empty volume string")
    low = s.lower()
    if low in ("mute", "-infdb"):
        return 0.0
    if low.endswith("db"):
        num_str = s[:-2].strip()
        try:
            db = float(num_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid dB volume: {val}") from exc
        if math.isnan(db) or db > 100:
            raise argparse.ArgumentTypeError(f"Invalid dB volume: {val}")
        return 0.0 if db == -math.inf else 10.0 ** (db / 20.0)
    if low.endswith("%"):
        num_str = s[:-1].strip()
        try:
            pct = float(num_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid percentage volume: {val}"
            ) from exc
        if math.isnan(pct) or pct < 0.0:
            raise argparse.ArgumentTypeError(
                f"Percentage volume cannot be negative: {val}"
            )
        return pct / 100.0
    try:
        num = float(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid volume value: {val}") from exc
    if math.isnan(num):
        raise argparse.ArgumentTypeError(f"Invalid volume value: {val}")
    if s.startswith("-"):
        if num == 0.0:
            raise argparse.ArgumentTypeError(f"Negative zero ratio not allowed: {val}")
        return 10.0 ** (num / 20.0)
    if s.startswith("+"):
        raise argparse.ArgumentTypeError(
            f"Explicit '+' only valid with dB suffix: {val}"
        )
    if num < 0.0:
        raise argparse.ArgumentTypeError(f"Negative ratio not allowed: {val}")
    return num


def format_duration(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def compute_fingerprint(files: list[Path], params: dict) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(sha256_file(p).encode("ascii"))
    h.update(json.dumps(params, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def load_manifest(workspace: Path) -> dict:
    p = workspace / "manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(workspace: Path, manifest: dict):
    p = workspace / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def check_stage_status(
    artifacts: list[Path],
    input_fp: str,
    manifest: dict,
    force: bool,
) -> tuple[bool, list[Path]]:
    """Returns (should_run, kept_edited_files)."""
    if force:
        return True, []

    edited = []
    missing = False
    stale = False

    for art in artifacts:
        entry = manifest.get(art.name)
        if not art.exists():
            missing = True
            continue
        if entry is None:
            edited.append(art)
            continue
        current_hash = sha256_file(art)
        if current_hash != entry.get("output_fp"):
            edited.append(art)
            if entry.get("input_fp") != input_fp:
                print(
                    f"  [warning] '{art.name}' was manually edited and its upstream "
                    f"inputs changed; keeping edited file (delete it or use --force to regenerate)."
                )
        elif entry.get("input_fp") != input_fp:
            stale = True

    if edited and not missing and not stale:
        return False, edited
    if not missing and not stale and not edited:
        return False, []
    return True, edited


def record_artifacts(
    artifacts: list[Path],
    kept_edited: list[Path],
    input_fp: str,
    manifest: dict,
    workspace: Path,
):
    kept_set = {p.name for p in kept_edited}
    for art in artifacts:
        if art.name in kept_set or not art.exists():
            continue
        manifest[art.name] = {
            "input_fp": input_fp,
            "output_fp": sha256_file(art),
        }
    save_manifest(workspace, manifest)


def verify_ffmpeg():
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit(
            "Error: 'ffmpeg' not found on PATH. Install FFmpeg with libass:\n"
            "  winget install Gyan.FFmpeg"
        )
    if not re.search(r"\bass\b", res.stdout):
        sys.exit(
            "Error: FFmpeg on PATH does not include the 'ass' (libass) filter.\n"
            "Install a full FFmpeg build with libass:\n"
            "  winget install Gyan.FFmpeg"
        )


def run_ffmpeg_with_progress(cmd: list[str], duration_sec: float, label: str):
    full_cmd = [
        cmd[0],
        "-hide_banner",
        "-nostats",
        "-progress",
        "pipe:1",
        *cmd[1:],
    ]
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr_lines: list[str] = []

    def _drain_stderr():
        for ln in proc.stderr:
            stderr_lines.append(ln)

    drain_thread = threading.Thread(target=_drain_stderr, daemon=True)
    drain_thread.start()

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us="):
            val_str = line.split("=", 1)[1]
            try:
                us = int(val_str)
                if duration_sec > 0:
                    pct = min(100.0, max(0.0, (us / 1e6) / duration_sec * 100.0))
                    print(f"\r  {label}: {pct:5.1f}%", end="", flush=True)
            except ValueError:
                pass
        elif line == "progress=end":
            print(f"\r  {label}: 100.0%", flush=True)

    ret = proc.wait()
    drain_thread.join(timeout=2.0)
    if ret != 0:
        stderr_out = "".join(stderr_lines)
        print(stderr_out, file=sys.stderr)
        raise RuntimeError(f"FFmpeg command failed with exit code {ret}")


def stage1_decode(song_path: Path, workspace: Path, manifest: dict, force: bool):
    source_wav = workspace / "source.wav"
    input_fp = compute_fingerprint([song_path], {"stage": 1, "sr": 44100, "ch": 2})
    should_run, kept = check_stage_status([source_wav], input_fp, manifest, force)
    if not should_run:
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(song_path),
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_f32le",
        str(source_wav),
    ]
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise RuntimeError("Stage 1 FFmpeg decode failed")

    record_artifacts([source_wav], kept, input_fp, manifest, workspace)
    return True


def stage2_separate(
    workspace: Path, sep_model: str, manifest: dict, force: bool
) -> bool:
    source_wav = workspace / "source.wav"
    vocals_wav = workspace / "vocals.wav"
    inst_wav = workspace / "instrumental.wav"
    artifacts = [vocals_wav, inst_wav]

    input_fp = compute_fingerprint([source_wav], {"stage": 2, "sep_model": sep_model})
    should_run, kept = check_stage_status(artifacts, input_fp, manifest, force)
    if not should_run:
        return False

    from audio_separator.separator import Separator

    model_dir = CACHE_DIR / "models" / "separator"
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = workspace / "_sep_tmp"
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True, exist_ok=True)

    sep = Separator(
        model_file_dir=str(model_dir),
        output_dir=str(tmp_out),
        output_format="WAV",
    )
    sep.load_model(model_filename=sep_model)
    out_files = sep.separate(str(source_wav))
    del sep
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    found_vocals = None
    found_inst = None
    for f in out_files:
        p = Path(f)
        if not p.is_absolute():
            p = tmp_out / p
        stem_lower = p.stem.lower()
        if "(vocals)" in stem_lower or "_vocals" in stem_lower:
            found_vocals = p
        elif (
            "(instrumental)" in stem_lower
            or "(other)" in stem_lower
            or "(no_vocals)" in stem_lower
            or "_instrumental" in stem_lower
        ):
            found_inst = p

    if found_vocals is None or found_inst is None:
        # Fallback inspection if labels differ
        for f in out_files:
            p = Path(f) if Path(f).is_absolute() else (tmp_out / f)
            if "vocal" in p.stem.lower() and found_vocals is None:
                found_vocals = p
            elif found_inst is None:
                found_inst = p

    if found_vocals is None or found_inst is None:
        raise RuntimeError(
            f"Could not identify vocal and instrumental stems from separator outputs: {out_files}"
        )

    kept_names = {p.name for p in kept}
    if vocals_wav.name not in kept_names:
        shutil.move(str(found_vocals), str(vocals_wav))
    if inst_wav.name not in kept_names:
        shutil.move(str(found_inst), str(inst_wav))
    shutil.rmtree(tmp_out, ignore_errors=True)

    record_artifacts(artifacts, kept, input_fp, manifest, workspace)
    return True


def prepare_lyrics_file(lyrics_arg: Path | None, workspace: Path):
    ws_lyrics = workspace / "lyrics.txt"
    if lyrics_arg is not None:
        supplied_bytes = lyrics_arg.read_bytes()
        if ws_lyrics.exists():
            existing_bytes = ws_lyrics.read_bytes()
            if existing_bytes != supplied_bytes:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                bak = workspace / f"lyrics.{ts}.bak.txt"
                ws_lyrics.rename(bak)
                print(f"  Backed up existing lyrics.txt to {bak.name}")
                shutil.copy2(lyrics_arg, ws_lyrics)
        else:
            shutil.copy2(lyrics_arg, ws_lyrics)


def strip_ruby_overrides(line: str) -> str:
    return re.sub(r"\{([^|{}]+)\|([^{}]+)\}", r"\1", line)


def normalized_characters(text):
    for index, char in enumerate(text):
        if char.isalnum():
            lower = char.lower()
            yield (lower if len(lower) == 1 else char), index


def map_alignment_chunks(lines, segments):
    """Map whole timed tokens onto lyric characters, ignoring segment boundaries."""
    positions = []
    reference = []
    for line_index, line in enumerate(lines):
        for char, index in normalized_characters(line):
            positions.append((line_index, index))
            reference.append(char)

    words = [word for segment in segments for word in (segment.words or ())]
    tokens = [
        "".join(char for char, _ in normalized_characters(word.word)) for word in words
    ]
    heard = "".join(tokens)
    expected = "".join(reference)
    if heard == expected:
        matched_positions = list(range(len(heard)))
    else:
        matched_positions = [None] * len(heard)
        matcher = difflib.SequenceMatcher(None, heard, expected, autojunk=False)
        for heard_start, ref_start, size in matcher.get_matching_blocks():
            matched_positions[heard_start : heard_start + size] = range(
                ref_start, ref_start + size
            )

    line_chunks = [[] for _ in lines]
    offset = 0
    for word, token in zip(words, tokens):
        matched = matched_positions[offset : offset + len(token)]
        offset += len(token)
        # A shared suffix of a different token is insufficient to time it.
        if (
            not matched
            or matched[0] is None
            or matched != list(range(matched[0], matched[0] + len(token)))
        ):
            continue
        if word.start is None or word.end is None:
            continue
        start, end = float(word.start), float(word.end)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            continue

        # A token can straddle a display line. Split only that token's interval.
        first = 0
        while first < len(matched):
            line_index, char_start = positions[matched[first]]
            stop = first + 1
            while stop < len(matched) and positions[matched[stop]][0] == line_index:
                stop += 1
            chunks = line_chunks[line_index]
            chunk_start = start + (end - start) * first / len(matched)
            chunk_end = start + (end - start) * stop / len(matched)
            if chunks:
                chunk_start = max(chunk_start, chunks[-1]["end"])
            if chunk_start < chunk_end:
                chunks.append(
                    {
                        "first": char_start,
                        "last": positions[matched[stop - 1]][1],
                        "start": chunk_start,
                        "end": chunk_end,
                    }
                )
            first = stop

    return [
        {"text": line, "chunks": chunks} for line, chunks in zip(lines, line_chunks)
    ]


def recover_untimed_lines(model, vocals_wav, language, lines, aligned):
    if all(line["chunks"] for line in aligned) and all(
        previous["chunks"][-1]["end"] <= current["chunks"][0]["start"] + 0.5
        for previous, current in zip(aligned, aligned[1:])
    ):
        return aligned

    print("  Searching the vocal transcription for untimed lyric lines.")
    transcript = model.transcribe(
        str(vocals_wav),
        language=language,
        vad=False,
        stream=False,
        fp16=False,
    )
    import whisper

    audio = whisper.load_audio(str(vocals_wav))
    recovered = list(aligned)
    for index, line in enumerate(lines):
        previous_end = max(
            (chunk["end"] for item in recovered[:index] for chunk in item["chunks"]),
            default=0.0,
        )
        if (
            recovered[index]["chunks"]
            and recovered[index]["chunks"][0]["start"] + 0.5 >= previous_end
        ):
            continue
        expected = "".join(char for char, _ in normalized_characters(line))
        if not expected:
            continue
        candidates = []
        for position, segment in enumerate(transcript.segments):
            if segment.start < previous_end:
                continue
            heard = segment.text
            end_segment = segment
            for following in transcript.segments[position : position + 2]:
                if following is not segment:
                    if following.start - segment.end > 4:
                        break
                    heard += following.text
                    end_segment = following
                normalized = "".join(char for char, _ in normalized_characters(heard))
                match = difflib.SequenceMatcher(
                    None, expected, normalized, autojunk=False
                )
                score = sum(block.size for block in match.get_matching_blocks()) / len(
                    expected
                )
                if score >= 0.7:
                    candidates.append((score, segment.start, end_segment.end))
        candidates.sort(key=lambda item: item[0], reverse=True)
        alternatives = (
            [
                candidate
                for candidate in candidates[1:]
                if candidate[1] >= candidates[0][2] or candidate[2] <= candidates[0][1]
            ]
            if candidates
            else []
        )
        if not candidates or (
            alternatives and candidates[0][0] - alternatives[0][0] < 0.1
        ):
            matches = ", ".join(
                f"{start:.1f}s ({score:.2f})" for score, start, _ in candidates[:3]
            )
            print(
                f"  [warning] Could not locate lyric line {index + 1} "
                f"unambiguously: {matches or 'no matches'}."
            )
            continue
        start = max(previous_end, candidates[0][1] - 1.0)
        end = min(len(audio) / 16000, candidates[0][2] + 1.0)
        result = model.align(
            audio[round(start * 16000) : round(end * 16000)],
            line,
            language=language,
            original_split=True,
            vad=False,
            nonspeech_skip=None,
        )
        if result is None:
            continue
        candidate = map_alignment_chunks([line], result.segments)[0]
        for chunk in candidate["chunks"]:
            chunk["start"] += start
            chunk["end"] += start
        if candidate["chunks"]:
            recovered[index] = candidate
            print(f"  Re-aligned lyric line {index + 1} at {start:.1f}s.")
    return recovered


def vocal_activity(samples, sample_rate):
    frame_size = max(1, round(sample_rate * 0.02))
    frame_seconds = frame_size / sample_rate
    frames = samples[: len(samples) // frame_size * frame_size].reshape(-1, frame_size)
    if not len(frames):
        return np.empty(0, dtype=bool), frame_seconds
    rms = np.sqrt(np.mean(frames**2, axis=1))
    peak = float(rms.max())
    floor = min(peak * 0.10, float(np.percentile(rms, 20)) * 3)
    return rms > max(peak * 0.02, floor), frame_seconds


def detect_audio_language(model, vocals_wav: Path) -> str:
    import whisper

    if not model.is_multilingual:
        return "en"

    audio = whisper.load_audio(str(vocals_wav))
    sample_rate = whisper.audio.SAMPLE_RATE
    active, frame_seconds = vocal_activity(audio, sample_rate)
    if not active.any():
        raise ValueError(
            "No vocal activity for language detection. Check vocals.wav or set --language."
        )

    # Sample throughout the song so an intro or a single chorus can't cast every vote.
    sample_count = min(5, max(1, round(len(active) * frame_seconds / 6)))
    window_frames = round(12 / frame_seconds)
    frame_samples = round(frame_seconds * sample_rate)
    totals = {}
    votes = []
    for region in np.array_split(np.arange(len(active)), sample_count):
        region_activity = active[region]
        window = min(window_frames, len(region))
        counts = np.convolve(
            region_activity.astype(int), np.ones(window, dtype=int), mode="valid"
        )
        best = int(counts.argmax())
        if counts[best] * frame_seconds < 0.5:
            continue
        first = int(region[best])
        audio_clip = audio[first * frame_samples : (first + window) * frame_samples]
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio_clip), model.dims.n_mels
        ).to(device=model.device, dtype=next(model.parameters()).dtype)
        _, probs = model.detect_language(mel)
        for lang, probability in probs.items():
            totals[lang] = totals.get(lang, 0.0) + probability
        votes.append(f"{first * frame_seconds:.1f}s {max(probs, key=probs.get)}")

    if not totals:
        raise ValueError(
            "Too little vocal activity for language detection. Set --language explicitly."
        )
    detected = max(totals, key=totals.get)
    leading = sorted(totals, key=totals.get, reverse=True)[:3]
    scores = ", ".join(f"{lang} {totals[lang] / len(votes):.0%}" for lang in leading)
    print(f"  Detected audio language: {detected} ({scores})")
    print("  Language by clip: " + ", ".join(votes))
    return detected


def stage3_align(
    workspace: Path,
    lyrics_arg: Path | None,
    whisper_model: str,
    language: str,
    manifest: dict,
    force: bool,
) -> bool:
    prepare_lyrics_file(lyrics_arg, workspace)
    vocals_wav = workspace / "vocals.wav"
    lyrics_txt = workspace / "lyrics.txt"
    alignment_json = workspace / "alignment.json"

    has_user_lyrics = lyrics_txt.exists() and (
        lyrics_arg is not None
        or lyrics_txt.name not in manifest
        or sha256_file(lyrics_txt) != manifest[lyrics_txt.name].get("output_fp")
    )

    if has_user_lyrics:
        upstream = [vocals_wav, lyrics_txt]
        artifacts = [alignment_json]
    else:
        upstream = [vocals_wav]
        artifacts = [lyrics_txt, alignment_json]

    input_fp = compute_fingerprint(
        upstream,
        {
            "stage": 3,
            "version": 4,
            "whisper_model": whisper_model,
            "language": language,
        },
    )
    should_run, kept = check_stage_status(artifacts, input_fp, manifest, force)
    if not should_run:
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_vram_gb = 0.0
    if device == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb < 6.0 and whisper_model.startswith("large"):
            print(
                f"  [warning] Whisper model '{whisper_model}' requires >6GB VRAM, but GPU 0 only has {total_vram_gb:.1f}GB. "
                "Falling back to CPU for Whisper."
            )
            device = "cpu"

    if device == "cpu":
        print(
            f"  [warning] Running Whisper '{whisper_model}' on CPU; this will be slow."
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    whisper_cache = CACHE_DIR / "models" / "whisper"
    whisper_cache.mkdir(parents=True, exist_ok=True)
    model = stable_whisper.load_model(
        whisper_model, device=device, download_root=str(whisper_cache)
    )

    resolved_lang = (
        language.lower() if (language and language.lower() != "auto") else None
    )

    if resolved_lang is None:
        resolved_lang = detect_audio_language(model, vocals_wav)

    if has_user_lyrics:
        raw_lines = [
            ln.strip()
            for ln in lyrics_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        clean_lines = [strip_ruby_overrides(ln) for ln in raw_lines]
        joined_text = "\n".join(clean_lines)

        if not any(char.isalnum() for char in joined_text):
            raise ValueError("The supplied lyrics file contains no lyric text.")

        result = model.align(
            str(vocals_wav),
            joined_text,
            language=resolved_lang,
            original_split=True,
            vad=True,
        )
        if result is None:
            raise ValueError(
                "Whisper could not align the supplied lyrics to vocals.wav."
            )
    else:
        transcribe_kwargs = {"language": resolved_lang, "stream": False}
        if device == "cuda" and total_vram_gb < 6.0:
            transcribe_kwargs["fp16"] = False
        # A speech VAD can classify sung phrases as silence even in an isolated stem.
        result = model.transcribe(str(vocals_wav), vad=False, **transcribe_kwargs)
        if not result.segments:
            print(
                "  [warning] No lyrics transcribed. Retrying with speech activity detection."
            )
            result = model.transcribe(str(vocals_wav), vad=True, **transcribe_kwargs)
        clean_lines = [seg.text.strip() for seg in result.segments if seg.text.strip()]
        if not any(char.isalnum() for line in clean_lines for char in line):
            raise ValueError(
                "Whisper returned no lyrics. Check vocals.wav or supply lyrics with --lyrics."
            )
        if lyrics_txt not in kept:
            lyrics_txt.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")

    out_lines = map_alignment_chunks(clean_lines, result.segments)
    if has_user_lyrics:
        out_lines = recover_untimed_lines(
            model, vocals_wav, resolved_lang, clean_lines, out_lines
        )
    if alignment_json not in kept:
        alignment_json.write_text(
            json.dumps({"lines": out_lines}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    record_artifacts(artifacts, kept, input_fp, manifest, workspace)
    return True


def katakana_to_hiragana(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def is_kanji(ch: str) -> bool:
    code = ord(ch)
    return (
        (0x4E00 <= code <= 0x9FFF)
        or (0x3400 <= code <= 0x4DBF)
        or (0xF900 <= code <= 0xFAFF)
        or ch == "々"
    )


def annotate_token_furigana(surface: str, kana_reading: str) -> str:
    if not any(is_kanji(c) for c in surface):
        return surface
    hira = katakana_to_hiragana(kana_reading) if kana_reading else ""
    if not hira:
        return surface

    s_Surf = katakana_to_hiragana(surface)
    prefix_len = 0
    while (
        prefix_len < len(surface)
        and prefix_len < len(hira)
        and not is_kanji(surface[prefix_len])
        and s_Surf[prefix_len] == hira[prefix_len]
    ):
        prefix_len += 1

    suffix_len = 0
    while (
        suffix_len < (len(surface) - prefix_len)
        and suffix_len < (len(hira) - prefix_len)
        and not is_kanji(surface[len(surface) - 1 - suffix_len])
        and s_Surf[len(surface) - 1 - suffix_len] == hira[len(hira) - 1 - suffix_len]
    ):
        suffix_len += 1

    prefix = surface[:prefix_len]
    suffix = surface[len(surface) - suffix_len :] if suffix_len > 0 else ""
    mid_surf = surface[prefix_len : len(surface) - suffix_len]
    mid_hira = hira[prefix_len : len(hira) - suffix_len]

    if not mid_surf or not mid_hira:
        return surface

    if all(is_kanji(c) for c in mid_surf):
        return f"{prefix}{{{mid_surf}|{mid_hira}}}{suffix}"

    # Mixed kanji and internal kana (e.g. 繰り返す -> くりかえす): match internal kana anchor
    kana_spans = [
        (m.start(), m.end(), katakana_to_hiragana(m.group()))
        for m in re.finditer(r"[^\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff々]+", mid_surf)
    ]
    if len(kana_spans) == 1:
        ks, ke, k_hira = kana_spans[0]
        pos = mid_hira.find(k_hira, 1)
        if 0 < pos and pos + len(k_hira) < len(mid_hira):
            left_k = mid_surf[:ks]
            right_k = mid_surf[ke:]
            left_r = mid_hira[:pos]
            right_r = mid_hira[pos + len(k_hira) :]
            return (
                f"{prefix}{{{left_k}|{left_r}}}{mid_surf[ks:ke]}"
                f"{{{right_k}|{right_r}}}{suffix}"
            )

    return f"{prefix}{{{mid_surf}|{mid_hira}}}{suffix}"


def generate_ruby_line(line: str, tagger: fugashi.Tagger) -> str:
    parts = re.split(r"(\{[^|{}]+\|[^{}]+\})", line)
    out = []
    for part in parts:
        if not part:
            continue
        if part.startswith("{") and part.endswith("}") and "|" in part:
            out.append(part)
        elif not any(0x4E00 <= ord(c) <= 0x9FFF for c in part):
            out.append(part)
        else:
            cursor = 0
            for word in tagger(part):
                surf = word.surface
                idx = part.find(surf, cursor)
                if idx != -1:
                    out.append(part[cursor:idx])
                    kana = getattr(word.feature, "kana", None) or ""
                    out.append(annotate_token_furigana(surf, kana))
                    cursor = idx + len(surf)
                else:
                    kana = getattr(word.feature, "kana", None) or ""
                    out.append(annotate_token_furigana(surf, kana))
            if cursor < len(part):
                out.append(part[cursor:])
    return "".join(out)


def split_morae(text: str) -> list[str]:
    morae = []
    for ch in text:
        if ch in NON_SUNG_CHARS:
            continue
        if ch in SMALL_KANA and morae:
            morae[-1] += ch
        else:
            morae.append(ch)
    return morae


def parse_ruby_line_segments(ruby_line: str) -> tuple[str, list[dict]]:
    """Parses a line with {base|ruby} into plain text and segment list."""
    pattern = re.compile(r"\{([^|{}]+)\|([^{}]+)\}")
    segments = []
    plain_chars = []
    pos = 0
    for m in pattern.finditer(ruby_line):
        if m.start() > pos:
            raw = ruby_line[pos : m.start()]
            start_idx = len(plain_chars)
            plain_chars.extend(list(raw))
            segments.append(
                {
                    "type": "plain",
                    "base": raw,
                    "ruby": None,
                    "first": start_idx,
                    "last": len(plain_chars) - 1,
                }
            )
        base_str = m.group(1)
        ruby_str = m.group(2)
        start_idx = len(plain_chars)
        plain_chars.extend(list(base_str))
        segments.append(
            {
                "type": "ruby",
                "base": base_str,
                "ruby": ruby_str,
                "first": start_idx,
                "last": len(plain_chars) - 1,
            }
        )
        pos = m.end()

    if pos < len(ruby_line):
        raw = ruby_line[pos:]
        start_idx = len(plain_chars)
        plain_chars.extend(list(raw))
        segments.append(
            {
                "type": "plain",
                "base": raw,
                "ruby": None,
                "first": start_idx,
                "last": len(plain_chars) - 1,
            }
        )

    return "".join(plain_chars), segments


def ensure_subtitle_font(font_arg: Path | None) -> Path:
    if font_arg is not None:
        if not font_arg.exists():
            sys.exit(f"Font file not found: {font_arg}")
        return font_arg.resolve()
    font_dir = CACHE_DIR / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_dir / DEFAULT_FONT_FILENAME
    if not font_path.exists():
        print(f"  Downloading subtitle font to {font_path}...")
        urllib.request.urlretrieve(DEFAULT_FONT_URL, str(font_path))
    return font_path


class FontMetrics:
    def __init__(self, font_path: Path):
        self.tt = TTFont(str(font_path))
        self.cmap = self.tt.getBestCmap()
        self.hmtx = self.tt["hmtx"].metrics
        os2 = self.tt["OS/2"]
        self.win_height = float(os2.usWinAscent + os2.usWinDescent)
        self.win_ascent = float(os2.usWinAscent)
        self.win_descent = float(os2.usWinDescent)
        self.family_name = self._extract_family_name()

    def _extract_family_name(self) -> str:
        name_table = self.tt["name"]
        for rec in name_table.names:
            if rec.nameID in (1, 4):
                try:
                    s = rec.toUnicode()
                    if s.isascii():
                        return s
                except UnicodeDecodeError:
                    pass
        return "Noto Sans CJK JP"

    def char_width(self, ch: str, font_size: float, scale_x: float = 1.0) -> float:
        glyph_name = self.cmap.get(ord(ch))
        if glyph_name and glyph_name in self.hmtx:
            adv, _ = self.hmtx[glyph_name]
        else:
            adv = int(self.win_height * 0.5)
        return (adv * font_size / self.win_height) * scale_x

    def text_width(self, text: str, font_size: float, scale_x: float = 1.0) -> float:
        return sum(self.char_width(c, font_size, scale_x) for c in text)


def format_ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    total_cs = int(round(sec * 100))
    h, rem = divmod(total_cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def compute_char_timings(
    plain_text: str,
    segments: list[dict],
    chunks: list[dict],
) -> list[tuple[float, float]]:
    n = len(plain_text)
    if n == 0:
        return []

    char_mora_weights = [0.0] * n
    for seg in segments:
        first, last = seg["first"], seg["last"]
        span_len = last - first + 1
        if seg["type"] == "ruby" and seg["ruby"]:
            morae = split_morae(seg["ruby"])
            w = max(1, len(morae)) / float(span_len)
            for idx in range(first, last + 1):
                char_mora_weights[idx] = w
        else:
            idx = first
            while idx <= last:
                ch = plain_text[idx]
                if ch in NON_SUNG_CHARS:
                    char_mora_weights[idx] = 0.0
                    idx += 1
                elif idx + 1 <= last and plain_text[idx + 1] in SMALL_KANA:
                    char_mora_weights[idx] = 0.5
                    char_mora_weights[idx + 1] = 0.5
                    idx += 2
                elif ch in SMALL_KANA:
                    char_mora_weights[idx] = 0.5
                    idx += 1
                else:
                    char_mora_weights[idx] = 1.0
                    idx += 1

    times = [(0.0, 0.0)] * n

    def distribute(first, stop, start, end):
        weights = char_mora_weights[first:stop]
        total = sum(weights)
        cursor = start
        for index, weight in enumerate(weights, first):
            duration = (end - start) * weight / total if total else 0.0
            times[index] = (cursor, cursor + duration)
            cursor += duration

    next_char = 0
    previous_end = chunks[0]["start"] if chunks else 0.0
    for chunk in chunks:
        distribute(next_char, chunk["first"], previous_end, chunk["start"])
        next_char = chunk["last"] + 1
        distribute(chunk["first"], next_char, chunk["start"], chunk["end"])
        previous_end = chunk["end"]
    distribute(next_char, n, previous_end, previous_end)
    return times


def validate_alignment(align_data, display_lines, duration):
    if not isinstance(align_data, list):
        raise ValueError("alignment.json must contain a list of lyric lines.")

    untimed = []
    partial = []
    for index, (line, plain) in enumerate(zip(align_data, display_lines), 1):
        if not isinstance(line, dict) or not isinstance(line.get("chunks"), list):
            raise ValueError(f"Invalid lyric line {index} in alignment.json.")
        if line.get("text") != plain:
            raise ValueError(
                f"Line {index} differs between alignment.json and lyrics_ruby.txt: "
                f"{line.get('text')!r} vs {plain!r}. Regenerate the stale file."
            )
        previous_char = -1
        previous_end = 0.0
        covered = set()
        for chunk in line["chunks"]:
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Invalid timing chunk in alignment.json line {index}."
                )
            first, last = chunk.get("first"), chunk.get("last")
            start, end = chunk.get("start"), chunk.get("end")
            if not (
                type(first) is int
                and type(last) is int
                and previous_char < first <= last < len(plain)
                and type(start) in (int, float)
                and type(end) in (int, float)
                and previous_end <= start < end <= duration + 0.01
            ):
                raise ValueError(
                    f"Invalid timing in alignment.json line {index}: {plain!r}, {chunk!r}"
                )
            covered.update(range(first, last + 1))
            previous_char, previous_end = last, end

        missing = sum(i not in covered for _, i in normalized_characters(plain))
        if missing:
            (partial if covered else untimed).append(str(index))

    if len(align_data) != len(display_lines):
        raise ValueError(
            "alignment.json and lyrics_ruby.txt have different line counts. "
            f"The first unpaired line is {min(len(align_data), len(display_lines)) + 1}."
        )
    if not any(line["chunks"] for line in align_data):
        raise ValueError(
            "No lyrics have usable timings. Check vocals.wav and the lyric text."
        )
    for label, affected in (
        ("Untimed lyric lines omitted from subtitles", untimed),
        ("Partly timed lyric lines", partial),
    ):
        if affected:
            print(
                f"  [warning] {label} ({len(affected)}), line numbers: "
                + ", ".join(affected[:5])
            )


def report_vocal_timing(align_data, samples, sample_rate):
    active, frame_seconds = vocal_activity(samples, sample_rate)
    uncovered = active.copy()
    silent_starts = []
    repeated_durations = []
    for line in align_data:
        chunks = line["chunks"]
        for chunk in chunks:
            start, end = chunk["start"], chunk["end"]
            first = max(0, math.floor((start - 0.12) / frame_seconds))
            last = math.ceil((end + 0.12) / frame_seconds)
            uncovered[first:last] = False
            if not active[first : math.ceil((start + 0.12) / frame_seconds)].any():
                silent_starts.append(f"{start:.2f}s")
        durations = [chunk["end"] - chunk["start"] for chunk in chunks]
        for first in range(len(durations) - 4):
            run = durations[first : first + 5]
            if max(run) - min(run) <= 0.01:
                repeated_durations.append(f"{chunks[first]['start']:.2f}s")
                break

    # Group brief pauses between syllables, and keep only sustained uncovered activity.
    vocal_frames = np.flatnonzero(uncovered)
    breaks = np.flatnonzero(np.diff(vocal_frames) * frame_seconds > 0.12) + 1
    vocal_gaps = [
        f"{run[0] * frame_seconds:.2f}-{(run[-1] + 1) * frame_seconds:.2f}s"
        for run in np.split(vocal_frames, breaks)
        if len(run) * frame_seconds >= 0.5
    ]
    for label, times in (
        ("Lyric chunks starting near vocal silence", silent_starts),
        ("Lines with repeated equal-length timings", repeated_durations),
        ("Vocal activity outside lyric timing", vocal_gaps),
    ):
        if times:
            print(f"  [warning] {label} ({len(times)}): " + ", ".join(times[:5]))


def stage4_ass(
    workspace: Path,
    font_path: Path,
    style_preset: str,
    manifest: dict,
    force: bool,
) -> bool:
    lyrics_txt = workspace / "lyrics.txt"
    alignment_json = workspace / "alignment.json"
    lyrics_ruby_txt = workspace / "lyrics_ruby.txt"
    karaoke_ass = workspace / "karaoke.ass"

    ruby_input_fp = compute_fingerprint([lyrics_txt], {"stage": "4_ruby"})
    run_ruby, kept_ruby = check_stage_status(
        [lyrics_ruby_txt], ruby_input_fp, manifest, force
    )

    if run_ruby:
        tagger = fugashi.Tagger()
        raw_lines = [
            ln.strip()
            for ln in lyrics_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        ruby_lines = [generate_ruby_line(ln, tagger) for ln in raw_lines]
        lyrics_ruby_txt.write_text("\n".join(ruby_lines) + "\n", encoding="utf-8")
        record_artifacts(
            [lyrics_ruby_txt], kept_ruby, ruby_input_fp, manifest, workspace
        )

    ass_input_fp = compute_fingerprint(
        [lyrics_ruby_txt, alignment_json, workspace / "vocals.wav", font_path],
        {"stage": "4_ass", "version": 2, "style": style_preset},
    )
    run_ass, kept_ass = check_stage_status([karaoke_ass], ass_input_fp, manifest, force)
    if not run_ass:
        return run_ruby

    align_data = json.loads(alignment_json.read_text(encoding="utf-8")).get("lines")
    ruby_lines = [
        ln.strip()
        for ln in lyrics_ruby_txt.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    display_lines = [parse_ruby_line_segments(line) for line in ruby_lines]
    samples, sample_rate = sf.read(
        workspace / "vocals.wav", dtype="float32", always_2d=True
    )
    validate_alignment(
        align_data, [plain for plain, _ in display_lines], len(samples) / sample_rate
    )
    report_vocal_timing(align_data, samples.mean(axis=1), sample_rate)
    metrics = FontMetrics(font_path)

    if style_preset == "joysound":
        unsung_fill = "&H00FFFFFF"
        unsung_out = "&H00402010"
        sung_fill = "&H0030D8FF"
        sung_out = "&H00003890"
        shad_col = "&H80000000"
    else:
        # DAM preset
        unsung_fill = "&H00FFFFFF"
        unsung_out = "&H00202020"
        sung_fill = "&H0024B6FF"
        sung_out = "&H000028B8"
        shad_col = "&H80000000"

    max_base_fs = 135.0
    min_base_fs = 72.0
    ruby_ratio = 0.45
    margin_x = 80.0
    safe_width = 1920.0 - 2.0 * margin_x
    slot_y = (400.0, 620.0)
    ruby_gap = 8.0

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2
Kerning: no

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BaseUnsung,{metrics.family_name},{int(max_base_fs)},{unsung_fill},{unsung_fill},{unsung_out},{shad_col},0,0,0,0,100,100,0,0,1,6,3,7,0,0,0,1
Style: BaseSung,{metrics.family_name},{int(max_base_fs)},{sung_fill},{sung_fill},{sung_out},{shad_col},0,0,0,0,100,100,0,0,1,6,3,7,0,0,0,1
Style: RubyUnsung,{metrics.family_name},{int(max_base_fs * ruby_ratio)},{unsung_fill},{unsung_fill},{unsung_out},{shad_col},0,0,0,0,100,100,0,0,1,3,2,7,0,0,0,1
Style: RubySung,{metrics.family_name},{int(max_base_fs * ruby_ratio)},{sung_fill},{sung_fill},{sung_out},{shad_col},0,0,0,0,100,100,0,0,1,3,2,7,0,0,0,1
Style: BreakTimer,{metrics.family_name},54,{unsung_fill},{unsung_fill},{unsung_out},{shad_col},0,0,0,0,100,100,0,0,1,4,2,5,0,0,0,1
Style: LeadInDot,{metrics.family_name},64,{sung_fill},{sung_fill},{sung_out},{shad_col},0,0,0,0,100,100,0,0,1,4,2,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    parsed_lines = []
    for (plain_text, segments), alignment in zip(display_lines, align_data):
        chunks = alignment["chunks"]
        if not chunks:
            continue
        char_times = compute_char_timings(plain_text, segments, chunks)
        t_start = chunks[0]["start"]
        t_end = max(c["end"] for c in chunks)
        parsed_lines.append(
            {
                "plain": plain_text,
                "segments": segments,
                "chunks": chunks,
                "char_times": char_times,
                "start": t_start,
                "end": t_end,
            }
        )

    # Compute slot assignment and display windows
    n_lines = len(parsed_lines)
    stanza_reset_gap = 2.5
    lead_in_gap = 5.0
    slot_reuse_hold = 0.3
    slot_reuse_blank = 0.35
    stanza_clear_blank = 0.4
    max_preview_lead = 2.5

    slot_idx = 0
    for i, item in enumerate(parsed_lines):
        gap_before = (
            (item["start"] - parsed_lines[i - 1]["end"]) if i > 0 else item["start"]
        )
        if i == 0 or gap_before >= stanza_reset_gap:
            slot_idx = 0
            item["is_stanza_start"] = True
        else:
            slot_idx = 1 - slot_idx
            item["is_stanza_start"] = False
        item["slot"] = slot_idx
        item["gap_before"] = gap_before
        item["is_lead_in"] = (i == 0 and item["start"] >= 2.0) or (
            i > 0 and gap_before >= lead_in_gap
        )

    # Determine disp_end first so disp_start can leave a visible blank gap after it
    for i, item in enumerate(parsed_lines):
        disp_end = item["end"] + 0.5
        has_same_slot_in_stanza = (
            i + 2 < n_lines
            and not parsed_lines[i + 1]["is_stanza_start"]
            and not parsed_lines[i + 2]["is_stanza_start"]
            and parsed_lines[i + 2]["slot"] == item["slot"]
        )
        if has_same_slot_in_stanza:
            next_same = parsed_lines[i + 2]
            avail = max(0.0, next_same["start"] - item["end"])
            hold = min(slot_reuse_hold, avail * 0.4)
            disp_end = min(disp_end, item["end"] + hold)
        elif i + 1 < n_lines and parsed_lines[i + 1]["is_stanza_start"]:
            next_stanza = parsed_lines[i + 1]
            avail = max(0.0, next_stanza["start"] - item["end"])
            hold = min(0.5, avail * 0.35)
            disp_end = min(disp_end, item["end"] + hold)

        item["disp_end"] = max(item["end"], disp_end)

    # Ensure the earlier slot of a finishing stanza never outlives the stanza's final line
    for i, item in enumerate(parsed_lines):
        if (
            i + 1 < n_lines
            and not parsed_lines[i + 1]["is_stanza_start"]
            and (i + 2 == n_lines or parsed_lines[i + 2]["is_stanza_start"])
        ):
            item["disp_end"] = min(item["disp_end"], parsed_lines[i + 1]["disp_end"])

    for i, item in enumerate(parsed_lines):
        if item["is_stanza_start"]:
            lead = 3.0 if item["is_lead_in"] else max_preview_lead
            disp_start = max(0.0, item["start"] - lead)
            if i > 0:
                prev_clear = parsed_lines[i - 1]["disp_end"] + stanza_clear_blank
                disp_start = min(item["start"], max(disp_start, prev_clear))
        elif parsed_lines[i - 1]["is_stanza_start"]:
            disp_start = parsed_lines[i - 1]["disp_start"]
        else:
            prev_same_slot = parsed_lines[i - 2]
            earliest = prev_same_slot["disp_end"] + slot_reuse_blank
            desired = max(earliest, item["start"] - max_preview_lead)
            disp_start = min(item["start"], desired)

        item["disp_start"] = disp_start

    events = []
    for item in parsed_lines:
        plain = item["plain"]
        segments = item["segments"]
        char_times = item["char_times"]
        slot = item["slot"]
        disp_s = item["disp_start"]
        disp_e = item["disp_end"]
        event_start_sec = int(round(disp_s * 100)) / 100.0
        event_end_sec = max(event_start_sec + 0.01, int(round(disp_e * 100)) / 100.0)
        ts_str = format_ass_time(event_start_sec)
        te_str = format_ass_time(event_end_sec)

        unit_w = metrics.text_width(plain, 1.0, 1.0)
        if unit_w > 0:
            fit_fs = safe_width / unit_w
            line_fs = min(max_base_fs, fit_fs)
            if line_fs < min_base_fs:
                line_fs = min_base_fs
                scale_x = safe_width / (unit_w * min_base_fs)
            else:
                scale_x = 1.0
        else:
            line_fs = max_base_fs
            scale_x = 1.0

        fscx_pct = int(round(scale_x * 100))
        fscx_tag = f"\\fscx{fscx_pct}" if fscx_pct != 100 else ""
        total_w = unit_w * line_fs * scale_x

        line_x = (1920.0 - total_w) * 0.5
        line_y = slot_y[slot]

        line_bord = round(line_fs * 0.065, 1)
        line_shad = round(line_fs * 0.035, 1)

        char_x = []
        cur_x = line_x
        for ch in plain:
            w = metrics.char_width(ch, line_fs, scale_x)
            char_x.append((cur_x, cur_x + w))
            cur_x += w

        fs_tag = f"\\fs{int(round(line_fs))}\\bord{line_bord:g}\\shad{line_shad:g}"
        pos_tag = f"{{\\an7\\pos({line_x:.2f},{line_y:.2f}){fs_tag}{fscx_tag}}}"
        events.append(
            f"Dialogue: 0,{ts_str},{te_str},BaseUnsung,,0,0,0,,{pos_tag}{plain}"
        )

        pad_base = (line_bord + line_shad) * scale_x + 2.0
        y0_base = int(math.floor(line_y - line_bord - line_shad - 4.0))
        y1_base = int(math.ceil(line_y + line_fs + line_bord + line_shad + 4.0))

        n_chars = len(plain)
        first_non_space = min(
            (i for i in range(n_chars) if not plain[i].isspace()), default=-1
        )
        last_non_space = max(
            (i for i in range(n_chars) if not plain[i].isspace()), default=-1
        )

        for c_idx in range(n_chars):
            if plain[c_idx].isspace():
                continue

            x_left, x_right = char_x[c_idx]
            clip_x0 = (
                int(math.floor(x_left - pad_base))
                if c_idx == first_non_space
                else int(round(x_left))
            )
            clip_x1 = (
                int(math.ceil(x_right + pad_base))
                if c_idx == last_non_space
                else int(round(x_right))
            )
            if clip_x1 <= clip_x0:
                continue

            ct0, ct1 = char_times[c_idx]
            raw_t0 = int(round((ct0 - event_start_sec) * 1000))
            raw_t1 = int(round((ct1 - event_start_sec) * 1000))
            rel_t0 = max(0, raw_t0)
            rel_t1 = max(rel_t0 + 1, raw_t1)

            wipe_tag = (
                f"{{\\an7\\pos({line_x:.2f},{line_y:.2f}){fs_tag}{fscx_tag}"
                f"\\clip({clip_x0},{y0_base},{clip_x0},{y1_base})"
                f"\\t({rel_t0},{rel_t1},\\clip({clip_x0},{y0_base},{clip_x1},{y1_base}))}}"
            )
            events.append(
                f"Dialogue: 1,{ts_str},{te_str},BaseSung,,0,0,0,,{wipe_tag}{plain}"
            )

        # Emit ruby layers
        ruby_fs = round(line_fs * ruby_ratio, 1)
        ruby_bord = round(ruby_fs * 0.07, 1)
        ruby_shad = round(ruby_fs * 0.04, 1)
        ruby_y = line_y - ruby_fs - ruby_gap
        y0_ruby = int(math.floor(ruby_y - ruby_bord - ruby_shad - 3.0))
        y1_ruby = int(math.ceil(ruby_y + ruby_fs + ruby_bord + ruby_shad + 3.0))
        pad_ruby = (ruby_bord + ruby_shad) * scale_x + 2.0

        has_ruby = any(seg["type"] == "ruby" and seg["ruby"] for seg in segments)
        item["line_x"] = line_x
        item["top_y"] = ruby_y if has_ruby else line_y

        for seg in segments:
            if seg["type"] != "ruby" or not seg["ruby"]:
                continue
            r_text = seg["ruby"]
            b_first, b_last = seg["first"], seg["last"]
            base_x0 = char_x[b_first][0]
            base_x1 = char_x[b_last][1]
            base_span_w = base_x1 - base_x0

            r_unscaled_w = metrics.text_width(r_text, ruby_fs, 1.0)
            r_scale_x = scale_x
            if r_unscaled_w * r_scale_x > base_span_w * 1.5 and r_unscaled_w > 0:
                r_scale_x = min(r_scale_x, (base_span_w * 1.5) / r_unscaled_w)
            r_fscx_pct = int(round(r_scale_x * 100))
            r_fscx_tag = f"\\fscx{r_fscx_pct}" if r_fscx_pct != 100 else ""
            r_total_w = r_unscaled_w * r_scale_x

            r_x = base_x0 + (base_span_w - r_total_w) * 0.5
            r_fs_tag = (
                f"\\fs{int(round(ruby_fs))}\\bord{ruby_bord:g}\\shad{ruby_shad:g}"
            )
            r_pos_tag = f"{{\\an7\\pos({r_x:.2f},{ruby_y:.2f}){r_fs_tag}{r_fscx_tag}}}"
            events.append(
                f"Dialogue: 0,{ts_str},{te_str},RubyUnsung,,0,0,0,,{r_pos_tag}{r_text}"
            )

            morae = split_morae(r_text)
            if not morae:
                morae = [r_text]
            n_morae = len(morae)

            g_t0 = char_times[b_first][0]
            g_t1 = char_times[b_last][1]
            g_dur = max(0.0, g_t1 - g_t0)

            cur_rx = r_x
            for m_idx, mora_str in enumerate(morae):
                mw = metrics.text_width(mora_str, ruby_fs, r_scale_x)
                mx0 = cur_rx
                mx1 = cur_rx + mw
                cur_rx = mx1

                r_clip0 = (
                    int(math.floor(mx0 - pad_ruby)) if m_idx == 0 else int(round(mx0))
                )
                r_clip1 = (
                    int(math.ceil(mx1 + pad_ruby))
                    if m_idx == n_morae - 1
                    else int(round(mx1))
                )
                if r_clip1 <= r_clip0:
                    continue

                mt0 = g_t0 + g_dur * (m_idx / n_morae)
                mt1 = g_t0 + g_dur * ((m_idx + 1) / n_morae)
                raw_mt0 = int(round((mt0 - event_start_sec) * 1000))
                raw_mt1 = int(round((mt1 - event_start_sec) * 1000))
                rel_mt0 = max(0, raw_mt0)
                rel_mt1 = max(rel_mt0 + 1, raw_mt1)

                r_wipe_tag = (
                    f"{{\\an7\\pos({r_x:.2f},{ruby_y:.2f}){r_fs_tag}{r_fscx_tag}"
                    f"\\clip({r_clip0},{y0_ruby},{r_clip0},{y1_ruby})"
                    f"\\t({rel_mt0},{rel_mt1},\\clip({r_clip0},{y0_ruby},{r_clip1},{y1_ruby}))}}"
                )
                events.append(
                    f"Dialogue: 1,{ts_str},{te_str},RubySung,,0,0,0,,{r_wipe_tag}{r_text}"
                )

    # Add instrumental break indicators (>10s gap) and visual lead-in cues (countdown dots)
    for i, item in enumerate(parsed_lines):
        prev_end = parsed_lines[i - 1]["end"] if i > 0 else 0.0
        gap_duration = item["start"] - prev_end
        lead_in_start = item["start"] - 3.0

        if gap_duration > 10.0:
            break_disp_start = (
                parsed_lines[i - 1]["disp_end"] + stanza_clear_blank if i > 0 else 0.0
            )
            break_disp_end = lead_in_start
            if break_disp_end > break_disp_start:
                # Render second-by-second countdown for the instrumental break.
                # Align interval bounds to whole seconds counting down to lead_in_start.
                t = break_disp_start
                pos_tag = r"{\pos(960,540)}"
                while t < break_disp_end:
                    # Time remaining until item["start"]
                    rem_sec = max(1, int(math.ceil(item["start"] - t)))
                    # The next step boundary is when remaining seconds changes
                    next_t = min(break_disp_end, item["start"] - (rem_sec - 1))
                    if next_t <= t:
                        next_t = min(break_disp_end, t + 1.0)
                    t0_str = format_ass_time(t)
                    t1_str = format_ass_time(next_t)
                    events.append(
                        f"Dialogue: 0,{t0_str},{t1_str},BreakTimer,,0,0,0,,{pos_tag}♪ 間奏 ({rem_sec})"
                    )
                    t = next_t

        if item.get("is_lead_in", False) and item["start"] >= 2.0:
            # 3-2-1 countdown dots above the left start of the upcoming line
            dot_y = item["top_y"] - 36.0
            base_x = item["line_x"] + 25.0
            for dot_idx in range(3):
                x_pos = base_x + dot_idx * 65.0
                d_t0 = max(0.0, item["start"] - (3.0 - dot_idx))
                d_t1 = item["start"]
                if d_t1 > d_t0:
                    t0_str = format_ass_time(d_t0)
                    t1_str = format_ass_time(d_t1)
                    pos_tag = rf"{{\pos({x_pos:.1f},{dot_y:.1f})}}"
                    events.append(
                        f"Dialogue: 0,{t0_str},{t1_str},LeadInDot,,0,0,0,,{pos_tag}●"
                    )

    karaoke_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    record_artifacts([karaoke_ass], kept_ass, ass_input_fp, manifest, workspace)
    return True


def read_guide_notes(midi_path: Path, duration_sec: float):
    mid = MidiFile(str(midi_path))
    notes = []
    cur_t = 0.0
    active: dict[int, float] = {}

    for msg in mid:
        cur_t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = cur_t
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                t0 = active.pop(msg.note)
                if cur_t > t0:
                    notes.append((t0, cur_t, msg.note))
    for note_num, t0 in active.items():
        if duration_sec > t0:
            notes.append((t0, duration_sec, note_num))

    return sorted(notes)


def synthesize_mallet_guide(
    midi_path: Path, out_wav: Path, duration_sec: float, timbre: str, sr: int = 44100
):
    notes = read_guide_notes(midi_path, duration_sec)
    total_samples = int(round(duration_sec * sr))
    buf = np.zeros(total_samples, dtype=np.float32)

    if timbre == "glockenspiel":
        partials = [
            (1.0, 1.0, 3.0),
            (2.76, 0.22, 8.0),
            (5.40, 0.08, 14.0),
        ]
    else:
        partials = [
            (1.0, 1.0, 2.2),
            (2.0, 0.25, 4.5),
            (4.0, 0.12, 9.0),
        ]

    attack_samples = max(1, int(round(0.014 * sr)))
    legato_attack_samples = max(1, int(round(0.028 * sr)))
    fade_samples = max(1, int(round(0.008 * sr)))
    tail_sec = 0.22

    for idx, (t0, t1, midi_note) in enumerate(notes):
        s0 = max(0, int(round(t0 * sr)))
        note_dur = max(0.05, t1 - t0)
        render_dur = min(duration_sec - t0, note_dur + tail_sec)
        if render_dur <= 0 or s0 >= total_samples:
            continue
        n_samp = min(total_samples - s0, int(round(render_dur * sr)))
        if n_samp <= 0:
            continue

        is_legato = idx > 0 and (t0 - notes[idx - 1][1]) <= 0.03
        overtone_scale = 0.35 if is_legato else 1.0

        t = np.arange(n_samp, dtype=np.float32) / float(sr)
        f0 = 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))
        # Subtle 5.5 Hz vibrato after 180 ms on sustained notes
        vib_env = np.clip((t - 0.18) / 0.15, 0.0, 1.0)
        phase = 2.0 * np.pi * f0 * t + (0.003 * f0) * vib_env * np.sin(
            2.0 * np.pi * 5.5 * t
        )
        tone = np.zeros(n_samp, dtype=np.float32)
        for ratio, amp, decay in partials:
            freq = f0 * ratio
            if freq < sr * 0.45:
                eff_amp = amp if ratio == 1.0 else amp * overtone_scale
                tone += eff_amp * np.sin(ratio * phase) * np.exp(-decay * t)

        env = np.ones(n_samp, dtype=np.float32)
        att_len = legato_attack_samples if is_legato else attack_samples
        att = min(att_len, n_samp)
        env[:att] = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, att, dtype=np.float32)))

        off_idx = min(n_samp, int(round(note_dur * sr)))
        if off_idx < n_samp:
            rel_len = n_samp - off_idx
            rel_curve = np.exp(-np.linspace(0.0, 5.5, rel_len, dtype=np.float32))
            if rel_len >= fade_samples:
                rel_curve[-fade_samples:] *= np.linspace(
                    1.0, 0.0, fade_samples, dtype=np.float32
                )
            env[off_idx:] *= rel_curve
        else:
            rel = min(fade_samples, n_samp)
            env[-rel:] *= np.linspace(1.0, 0.0, rel, dtype=np.float32)

        buf[s0 : s0 + n_samp] += tone * env

    peak = float(np.max(np.abs(buf)))
    if peak > 1e-6:
        target_peak = 10.0 ** (-1.0 / 20.0)
        buf *= target_peak / peak

    stereo = np.column_stack([buf, buf])
    sf.write(str(out_wav), stereo, sr, subtype="FLOAT")


def synthesize_continuous_guide(
    midi_path: Path,
    vocals_wav: Path,
    out_wav: Path,
    duration_sec: float,
    sr: int = 44100,
):
    buf = np.zeros(int(round(duration_sec * sr)), dtype=np.float32)
    vocals, vocal_rate = sf.read(vocals_wav, dtype="float32", always_2d=True)
    hop = max(1, round(vocal_rate * 0.01))
    power = np.mean(vocals**2, axis=1)
    frames = np.pad(power, (0, -len(power) % hop)).reshape(-1, hop)
    rms = np.sqrt(frames.mean(axis=1))
    if len(rms) >= 3:
        rms = np.convolve(rms, [0.25, 0.5, 0.25], mode="same")
    if len(rms):
        reference = max(float(np.percentile(rms, 95)), 1e-6)
        # Compress the vocal dynamics so quiet syllables remain audible in the mix.
        envelope = np.clip(rms / reference, 0.0, 1.0) ** 0.6
        frame_times = (np.arange(len(rms)) + 0.5) * hop / vocal_rate
    for start, end, pitch in read_guide_notes(midi_path, duration_sec):
        first = max(0, round(start * sr))
        stop = min(len(buf), round(end * sr))
        length = stop - first
        if length <= 0:
            continue
        phase = np.arange(length) * (2 * np.pi * 440 / sr) * 2 ** ((pitch - 69) / 12)
        tone = np.sin(phase) + 0.22 * np.sin(2 * phase) + 0.06 * np.sin(3 * phase)
        if len(rms):
            tone *= 0.12 + 0.88 * np.interp(
                np.arange(first, stop) / sr, frame_times, envelope
            )
        fade_length = min(round(0.018 * sr), length // 2)
        if fade_length:
            fade = np.sin(np.linspace(0, np.pi / 2, fade_length)) ** 2
            tone[:fade_length] *= fade
            tone[-fade_length:] *= fade[::-1]
        buf[first:stop] += tone

    peak = float(np.max(np.abs(buf), initial=0.0))
    if peak > 1e-6:
        buf *= 10 ** (-1 / 20) / peak
    sf.write(str(out_wav), np.column_stack([buf, buf]), sr, subtype="FLOAT")


def decode_guide_pitch(probabilities):
    bins, _ = torchcrepe.decode.viterbi(probabilities)
    # Use CREPE's bin centers without the random dither in its Hz conversion.
    cents = torchcrepe.CENTS_PER_BIN * bins + 1997.3794084376191
    return bins, torchcrepe.convert.cents_to_frequency(cents)


def octave_harmonic_ratio(audio, sample_rate, start, end, lower_pitch):
    lower_hz = 440 * 2 ** ((lower_pitch - 69) / 12)
    fft_size = 4096
    ratios = []
    for fraction in (0.25, 0.5, 0.75):
        center = start + (end - start) * fraction
        first = round(max(start, center - 0.05) * sample_rate)
        last = round(min(end, center + 0.05) * sample_rate)
        window = audio[first:last]
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window)), n=fft_size))
        powers = []
        for harmonic in range(1, 7):
            bin_index = round(harmonic * lower_hz * fft_size / sample_rate)
            radius = max(2, round(0.025 * bin_index))
            band = spectrum[max(0, bin_index - radius) : bin_index + radius + 1]
            powers.append(np.max(band, initial=0.0) ** 2)
        even = sum(powers[1::2])
        if even > 0:
            ratios.append(sum(powers[::2]) / even)
    return np.median(ratios) if len(ratios) >= 2 else None


def correct_guide_octaves(notes, audio, sample_rate, lyric_starts):
    """Resolve octave ambiguity within a phrase using the stem's harmonics."""
    corrected = []
    for start, end, pitch in notes:
        if not corrected:
            corrected.append((start, end, pitch))
            continue

        previous_start, previous_end, previous_pitch = corrected[-1]
        phrase = np.searchsorted(lyric_starts, start, side="right")
        previous_phrase = np.searchsorted(lyric_starts, previous_start, side="right")
        delta = pitch - previous_pitch
        stop = min(end, lyric_starts[phrase]) if phrase < len(lyric_starts) else end
        if (
            phrase != previous_phrase
            or start - previous_end > 0.25
            or not 10 <= abs(delta) <= 14
            or stop - start < 0.10
        ):
            corrected.append((start, end, pitch))
            continue

        alternate = pitch - 12 if delta > 0 else pitch + 12
        # The even harmonics also fit the upper octave. Strong odd
        # harmonics distinguish the lower fundamental from its octave.
        ratio = octave_harmonic_ratio(
            audio, sample_rate, start, stop, min(pitch, alternate)
        )
        if ratio is None or not (
            (delta < 0 and ratio < 0.32) or (delta > 0 and ratio > 1.25)
        ):
            corrected.append((start, end, pitch))
            continue

        corrected.append((start, stop, alternate))
        if stop < end:
            corrected.append((stop, end, pitch))
    return corrected


def guide_pitch_notes(pitches, periodicity):
    notes = []
    start = 0
    pitch_start = 0
    end = 0

    def stable_short_note(first, stop, pitch):
        values = pitches[first:stop]
        return (
            len(values) >= 7
            and np.all(values > 0)
            and np.count_nonzero(np.abs(values - pitch) <= 0.65) >= len(values) - 1
            and np.median(periodicity[first:stop]) >= 0.65
        )

    def finish_note(stop):
        pitch = round(float(np.median(pitches[pitch_start:stop])))
        if stop - pitch_start >= 10 or stable_short_note(pitch_start, stop, pitch):
            notes.append((start * 0.01, stop * 0.01, pitch))
            return True
        return False

    while end < len(pitches):
        if pitches[end] <= 0:
            if end > pitch_start:
                finish_note(end)
            end += 1
            start = pitch_start = end
            continue

        if end > pitch_start:
            center = float(np.median(pitches[pitch_start:end]))
            if abs(pitches[end] - center) >= 0.75 and round(
                float(pitches[end])
            ) != round(center):
                ahead = pitches[end : end + 12]
                short_ahead = ahead[:7]
                target = round(float(np.median(short_ahead)))
                holds = (
                    len(ahead) == 12
                    and np.all(ahead > 0)
                    and np.count_nonzero(np.abs(ahead - target) <= 0.8) >= 10
                )
                if (
                    target > 0
                    and target != round(center)
                    and (
                        holds or stable_short_note(end, end + len(short_ahead), target)
                    )
                ):
                    if finish_note(end):
                        start = end
                    pitch_start = end
        end += 1

    if end > pitch_start:
        finish_note(end)
    return notes


def stage5_guide(
    workspace: Path,
    guide_timbre: str,
    manifest: dict,
    force: bool,
) -> bool:
    source_wav = workspace / "source.wav"
    vocals_wav = workspace / "vocals.wav"
    alignment_json = workspace / "alignment.json"
    guide_mid = workspace / "guide.mid"
    guide_wav = workspace / "guide.wav"

    mid_fp = compute_fingerprint(
        [vocals_wav, alignment_json],
        {"stage": "5_mid_v6", "timbre": guide_timbre},
    )
    run_mid, kept_mid = check_stage_status([guide_mid], mid_fp, manifest, force)

    if run_mid:
        audio, sr = sf.read(str(vocals_wav), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        target_sr = 16000
        if sr != target_sr:
            g = math.gcd(sr, target_sr)
            mono_16k = resample_poly(mono, target_sr // g, sr // g).astype(np.float32)
        else:
            mono_16k = mono

        hop_length = 160  # 10 ms at 16 kHz
        device = "cuda" if torch.cuda.is_available() else "cpu"
        crepe_model = "full" if device == "cuda" else "tiny"
        audio_tensor = torch.from_numpy(mono_16k).unsqueeze(0)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        f0, pd = torchcrepe.predict(
            audio_tensor,
            sample_rate=target_sr,
            hop_length=hop_length,
            fmin=45.0,
            fmax=1100.0,
            model=crepe_model,
            decoder=decode_guide_pitch,
            return_periodicity=True,
            batch_size=1024 if device == "cuda" else 256,
            device=device,
        )
        f0_np = f0.squeeze(0).cpu().numpy()
        pd_np = pd.squeeze(0).cpu().numpy()
        del audio_tensor, f0, pd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n_frames = len(f0_np)

        rms = np.zeros(n_frames, dtype=np.float32)
        win = hop_length * 2
        for i in range(n_frames):
            s = i * hop_length
            chunk = mono_16k[s : s + win]
            if len(chunk) > 0:
                rms[i] = float(np.sqrt(np.mean(chunk * chunk)))

        peak_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
        rms_gate = peak_rms * 0.03
        voiced = (pd_np > 0.5) & (rms > rms_gate) & (f0_np >= 45.0)

        align_data = json.loads(alignment_json.read_text(encoding="utf-8")).get(
            "lines", []
        )
        chunks = [chunk for line in align_data for chunk in line.get("chunks", [])]
        if chunks:
            lyric_mask = np.zeros(n_frames, dtype=bool)
            for chunk in chunks:
                start = max(0, math.floor((chunk["start"] - 0.12) / 0.01))
                stop = min(n_frames, math.ceil((chunk["end"] + 0.12) / 0.01))
                lyric_mask[start:stop] = True
            voiced &= lyric_mask

        valid_f0 = np.maximum(f0_np, 1e-3)
        raw_midi = 69.0 + 12.0 * np.log2(valid_f0 / 440.0)

        # Reconnect held pitches when CREPE loses confidence through a consonant.
        # Require energy through the gap so this does not extend notes across breaths.
        voiced_bridged = voiced.copy()
        midi_bridged = raw_midi.copy()
        idx = 0
        while idx < n_frames:
            if not voiced_bridged[idx]:
                end_idx = idx
                while end_idx < n_frames and not voiced_bridged[end_idx]:
                    end_idx += 1
                gap_len = end_idx - idx
                if idx >= 3 and end_idx + 3 <= n_frames and gap_len <= 14:
                    before = slice(idx - 3, idx)
                    after = slice(end_idx, end_idx + 3)
                    p_prev = float(np.median(raw_midi[before]))
                    p_next = float(np.median(raw_midi[after]))
                    min_energy = 0.25 * min(
                        np.median(rms[before]), np.median(rms[after])
                    )
                    if (
                        voiced[before].all()
                        and voiced[after].all()
                        and abs(p_next - p_prev) <= 0.8
                        and np.min(rms[idx:end_idx]) > max(rms_gate, min_energy)
                    ):
                        voiced_bridged[idx:end_idx] = True
                        midi_bridged[idx:end_idx] = np.linspace(
                            p_prev, p_next, gap_len + 2
                        )[1:-1]
                idx = end_idx
            else:
                idx += 1

        # Median filter within contiguous voiced segments so 0.0 doesn't pull boundaries down
        midi_filt = np.zeros(n_frames, dtype=np.float32)
        idx = 0
        while idx < n_frames:
            if voiced_bridged[idx]:
                end_idx = idx
                while end_idx < n_frames and voiced_bridged[end_idx]:
                    end_idx += 1
                seg = midi_bridged[idx:end_idx]
                if len(seg) >= 7:
                    midi_filt[idx:end_idx] = median_filter(seg, size=7, mode="nearest")
                elif len(seg) >= 3:
                    midi_filt[idx:end_idx] = median_filter(seg, size=3, mode="nearest")
                else:
                    midi_filt[idx:end_idx] = seg
                idx = end_idx
            else:
                idx += 1

        raw_notes = guide_pitch_notes(midi_filt, pd_np)

        notes = []
        idx = 0
        while idx < len(raw_notes):
            if idx + 2 < len(raw_notes):
                t0_a, t1_a, p_a = raw_notes[idx]
                t0_b, t1_b, p_b = raw_notes[idx + 1]
                t0_c, t1_c, p_c = raw_notes[idx + 2]
                if (
                    p_a == p_c
                    and abs(p_b - p_a) == 12
                    and (t1_b - t0_b) < 0.20
                    and (t0_b - t1_a) <= 0.05
                    and (t0_c - t1_b) <= 0.05
                ):
                    raw_notes[idx + 2] = (t0_a, t1_c, p_a)
                    idx += 2
                    continue
            notes.append(raw_notes[idx])
            idx += 1

        lyric_starts = sorted(
            min(chunk["start"] for chunk in line["chunks"])
            for line in align_data
            if line.get("chunks")
        )
        if lyric_starts:
            notes = correct_guide_octaves(notes, mono_16k, target_sr, lyric_starts)

        mid = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        mid.tracks.append(track)
        tempo = 500000
        track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
        program = 9 if guide_timbre == "glockenspiel" else 11
        track.append(Message("program_change", program=program, time=0))

        last_tick = 0
        for t0, t1, pitch in notes:
            on_tick = round(mido.second2tick(t0, mid.ticks_per_beat, tempo))
            off_tick = round(mido.second2tick(t1, mid.ticks_per_beat, tempo))
            track.append(
                Message("note_on", note=pitch, velocity=100, time=on_tick - last_tick)
            )
            track.append(
                Message("note_off", note=pitch, velocity=0, time=off_tick - on_tick)
            )
            last_tick = off_tick

        mid.save(str(guide_mid))
        record_artifacts([guide_mid], kept_mid, mid_fp, manifest, workspace)

    wav_inputs = [guide_mid, source_wav]
    if guide_timbre == "continuous":
        wav_inputs.append(vocals_wav)
    wav_fp = compute_fingerprint(
        wav_inputs, {"stage": "5_wav_v5", "timbre": guide_timbre}
    )
    run_wav, kept_wav = check_stage_status([guide_wav], wav_fp, manifest, force)
    if not run_wav:
        return run_mid

    info = sf.info(str(source_wav))
    if guide_timbre == "continuous":
        synthesize_continuous_guide(
            guide_mid,
            vocals_wav,
            guide_wav,
            info.duration,
            sr=info.samplerate,
        )
    else:
        synthesize_mallet_guide(
            guide_mid, guide_wav, info.duration, guide_timbre, sr=info.samplerate
        )
    record_artifacts([guide_wav], kept_wav, wav_fp, manifest, workspace)
    return True


def stage6_background(bg_arg: Path | None) -> Path:
    if bg_arg is not None:
        if not bg_arg.exists():
            sys.exit(f"Background file not found: {bg_arg}")
        return bg_arg.resolve()

    bg_dir = CACHE_DIR / "bg"
    bg_dir.mkdir(parents=True, exist_ok=True)
    default_bg = bg_dir / "bg_default_1080p30.mp4"
    if default_bg.exists():
        return default_bg

    w, h, fps, n_frames = 1920, 1080, 30, 300
    rng = np.random.default_rng(42)
    n_particles = 55

    x0 = rng.uniform(0, w, n_particles)
    y0 = rng.uniform(0, h, n_particles)
    vy_cycles = rng.choice([-2, -1, 1], size=n_particles)
    vx_amp = rng.uniform(18.0, 65.0, n_particles)
    vx_cycles = rng.integers(1, 3, size=n_particles)
    vx_phase = rng.uniform(0, 2.0 * np.pi, n_particles)
    tw_cycles = rng.integers(1, 4, size=n_particles)
    tw_phase = rng.uniform(0, 2.0 * np.pi, n_particles)
    sigmas = rng.uniform(10.0, 34.0, n_particles)

    palette = np.array(
        [
            [90, 160, 255],
            [160, 120, 255],
            [80, 220, 240],
            [255, 190, 120],
        ],
        dtype=np.float32,
    )
    colors = palette[rng.integers(0, len(palette), size=n_particles)]

    y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    top_col = np.array([10, 14, 30], dtype=np.float32)
    bot_col = np.array([24, 14, 44], dtype=np.float32)
    grad_row = (1.0 - y_norm) * top_col + y_norm * bot_col
    base_frame = np.broadcast_to(grad_row[:, None, :], (h, w, 3)).copy()

    stamps = []
    for p in range(n_particles):
        sig = float(sigmas[p])
        r = int(math.ceil(sig * 3.0))
        gy, gx = np.mgrid[-r : r + 1, -r : r + 1]
        kernel = np.exp(-(gx * gx + gy * gy) / (2.0 * sig * sig)).astype(np.float32)
        stamps.append((r, kernel[:, :, None] * colors[p][None, None, :]))

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(default_bg),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    assert proc.stderr is not None

    for f_idx in range(n_frames):
        tau = f_idx / float(n_frames)
        frame = base_frame.astype(np.float32)

        px = (x0 + vx_amp * np.sin(2.0 * np.pi * vx_cycles * tau + vx_phase)) % w
        py = (y0 + vy_cycles * h * tau) % h
        tw = 0.25 + 0.55 * (
            0.5 + 0.5 * np.sin(2.0 * np.pi * tw_cycles * tau + tw_phase)
        )

        for p in range(n_particles):
            cx, cy = int(round(float(px[p]))), int(round(float(py[p])))
            r, stamp = stamps[p]
            alpha = float(tw[p])

            y_min, y_max = max(0, cy - r), min(h, cy + r + 1)
            x_min, x_max = max(0, cx - r), min(w, cx + r + 1)
            if y_min >= y_max or x_min >= x_max:
                continue
            sy0, sy1 = y_min - (cy - r), y_max - (cy - r)
            sx0, sx1 = x_min - (cx - r), x_max - (cx - r)
            frame[y_min:y_max, x_min:x_max] += stamp[sy0:sy1, sx0:sx1] * alpha

        rgb = np.clip(frame, 0, 255).astype(np.uint8)
        proc.stdin.write(rgb.tobytes())
        if (f_idx + 1) % 30 == 0 or f_idx == n_frames - 1:
            print(
                f"\r  Generating default background: {f_idx + 1}/{n_frames} frames",
                end="",
                flush=True,
            )

    print()
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"Default background encode failed: {err}")

    return default_bg


def escape_ffmpeg_filter_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    s = s.replace(":", "\\:").replace("'", "\\'")
    return f"'{s}'"


def stage7_render(
    workspace: Path,
    song_stem: str,
    bg_path: Path,
    font_path: Path,
    audio_layout: str,
    inst_vol: float,
    guide_vol: float,
    vocal_vol: float,
    manifest: dict,
    force: bool,
) -> Path:
    source_wav = workspace / "source.wav"
    inst_wav = workspace / "instrumental.wav"
    guide_wav = workspace / "guide.wav"
    vocals_wav = workspace / "vocals.wav"
    karaoke_ass = workspace / "karaoke.ass"
    output_mp4 = workspace / f"{song_stem}_karaoke.mp4"

    use_guide = guide_vol > 0.0
    upstream_files = (
        [inst_wav, guide_wav, vocals_wav, karaoke_ass, bg_path, font_path]
        if use_guide
        else [inst_wav, vocals_wav, karaoke_ass, bg_path, font_path]
    )
    input_fp = compute_fingerprint(
        upstream_files,
        {
            "stage": 7,
            "audio_layout": audio_layout,
            "inst_vol": inst_vol,
            "guide_vol": guide_vol,
            "vocal_vol": vocal_vol,
        },
    )
    should_run, kept = check_stage_status([output_mp4], input_fp, manifest, force)
    if not should_run:
        return output_mp4

    duration_sec = sf.info(str(source_wav)).duration
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if bg_path.suffix.lower() in image_exts:
        bg_in = ["-loop", "1", "-framerate", "30", "-i", str(bg_path)]
    else:
        bg_in = ["-stream_loop", "-1", "-i", str(bg_path)]

    ass_esc = escape_ffmpeg_filter_path(karaoke_ass)
    fontsdir_esc = escape_ffmpeg_filter_path(font_path.parent)
    v_filter = (
        f"[0:v]scale=1920:1080:force_original_aspect_ratio=increase,"
        f"crop=1920:1080,fps=30,ass={ass_esc}:fontsdir={fontsdir_esc}[vout]"
    )

    if use_guide:
        audio_inputs = [
            "-i",
            str(inst_wav),
            "-i",
            str(guide_wav),
            "-i",
            str(vocals_wav),
        ]
        if audio_layout == "mix":
            filter_complex = (
                f"{v_filter};"
                f"[1:a]volume={inst_vol}[a_inst];"
                f"[2:a]volume={guide_vol}[a_guide];"
                f"[3:a]volume={vocal_vol}[a_voc];"
                f"[a_inst][a_guide][a_voc]amix=inputs=3:normalize=0,"
                f"alimiter=limit=0.97[aout]"
            )
            map_args = ["-map", "[vout]", "-map", "[aout]"]
        elif audio_layout == "dual-track":
            filter_complex = (
                f"{v_filter};"
                f"[1:a]volume={inst_vol},alimiter=limit=0.97[a_off];"
                f"[1:a]volume={inst_vol}[a_inst];"
                f"[2:a]volume={guide_vol}[a_guide];"
                f"[3:a]volume={vocal_vol}[a_voc];"
                f"[a_inst][a_guide][a_voc]amix=inputs=3:normalize=0,"
                f"alimiter=limit=0.97[a_guide_mix]"
            )
            map_args = [
                "-map",
                "[vout]",
                "-map",
                "[a_off]",
                "-map",
                "[a_guide_mix]",
                "-metadata:s:a:0",
                "title=Off vocal",
                "-metadata:s:a:1",
                "title=Guide",
                "-disposition:a:0",
                "default",
            ]
        elif audio_layout == "split-lr":
            filter_complex = (
                f"{v_filter};"
                f"[1:a]volume={inst_vol}[a_inst];"
                f"[2:a]volume={guide_vol}[a_guide];"
                f"[a_inst][a_guide]amix=inputs=2:normalize=0,"
                f"pan=mono|c0=0.5*c0+0.5*c1[a_left];"
                f"[3:a]volume={vocal_vol},pan=mono|c0=0.5*c0+0.5*c1[a_right];"
                f"[a_left][a_right]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR,"
                f"alimiter=limit=0.97[aout]"
            )
            map_args = ["-map", "[vout]", "-map", "[aout]"]
        else:
            raise ValueError(f"Unsupported audio layout: {audio_layout}")
    else:
        audio_inputs = ["-i", str(inst_wav), "-i", str(vocals_wav)]
        if audio_layout == "mix":
            filter_complex = (
                f"{v_filter};"
                f"[1:a]volume={inst_vol}[a_inst];"
                f"[2:a]volume={vocal_vol}[a_voc];"
                f"[a_inst][a_voc]amix=inputs=2:normalize=0,"
                f"alimiter=limit=0.97[aout]"
            )
            map_args = ["-map", "[vout]", "-map", "[aout]"]
        elif audio_layout == "dual-track":
            filter_complex = (
                f"{v_filter};"
                f"[1:a]volume={inst_vol},alimiter=limit=0.97[a_off];"
                f"[1:a]volume={inst_vol}[a_inst];"
                f"[2:a]volume={vocal_vol}[a_voc];"
                f"[a_inst][a_voc]amix=inputs=2:normalize=0,"
                f"alimiter=limit=0.97[a_guide_mix]"
            )
            map_args = [
                "-map",
                "[vout]",
                "-map",
                "[a_off]",
                "-map",
                "[a_guide_mix]",
                "-metadata:s:a:0",
                "title=Off vocal",
                "-metadata:s:a:1",
                "title=Guide",
                "-disposition:a:0",
                "default",
            ]
        elif audio_layout == "split-lr":
            filter_complex = (
                f"{v_filter};"
                f"[1:a]volume={inst_vol},pan=mono|c0=0.5*c0+0.5*c1[a_left];"
                f"[2:a]volume={vocal_vol},pan=mono|c0=0.5*c0+0.5*c1[a_right];"
                f"[a_left][a_right]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR,"
                f"alimiter=limit=0.97[aout]"
            )
            map_args = ["-map", "[vout]", "-map", "[aout]"]
        else:
            raise ValueError(f"Unsupported audio layout: {audio_layout}")

    cmd = [
        "ffmpeg",
        "-y",
        *bg_in,
        *audio_inputs,
        "-filter_complex",
        filter_complex,
        *map_args,
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_mp4),
    ]
    run_ffmpeg_with_progress(cmd, duration_sec, "Encoding MP4")
    record_artifacts([output_mp4], kept, input_fp, manifest, workspace)
    return output_mp4


def resolve_workspace(workspace_arg: str | None, song_stem: str) -> Path:
    if workspace_arg is None:
        ws = (REPO_DIR / "songs" / f"{song_stem}_karaoke").resolve()
    else:
        p = Path(workspace_arg)
        ws = (p if p.is_absolute() else (REPO_DIR / p)).resolve()
    try:
        ws.relative_to(REPO_DIR)
    except ValueError:
        sys.exit(f"Error: --workspace must resolve inside the repo ({REPO_DIR}): {ws}")
    return ws


def check_forced_edits(workspace: Path, force_idx: int, manifest: dict):
    stage_artifacts = [
        ["source.wav"],
        ["vocals.wav", "instrumental.wav"],
        ["lyrics.txt", "alignment.json"],
        ["lyrics_ruby.txt", "karaoke.ass"],
        ["guide.mid", "guide.wav"],
        [],
        [],
    ]
    edited = []
    for idx in range(force_idx, len(stage_artifacts)):
        for name in stage_artifacts[idx]:
            p = workspace / name
            if p.exists() and name in manifest:
                if sha256_file(p) != manifest[name].get("output_fp"):
                    edited.append(name)
    if edited:
        ans = input(
            f"Warning: --force will discard manual edits in {', '.join(edited)}. Continue? [y/N]: "
        )
        if ans.strip().lower() not in ("y", "yes"):
            sys.exit("Aborted.")


def main():
    parser = argparse.ArgumentParser(
        description="Japanese arcade-style karaoke video generator"
    )
    parser.add_argument("input", help="Song audio file")
    parser.add_argument("-l", "--lyrics", default=None, help="Lyrics text file")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Song workspace folder (inside the repo)",
    )
    parser.add_argument("--bg", default=None, help="Background video or image")
    parser.add_argument(
        "--audio-layout",
        choices=["mix", "dual-track", "split-lr"],
        default="mix",
        help="Audio stream layout",
    )
    parser.add_argument(
        "--inst-vol",
        type=parse_volume,
        default=1.0,
        help="Instrumental volume level",
    )
    parser.add_argument(
        "--guide-vol",
        type=parse_volume,
        default=0.3,
        help="Guide chime volume level",
    )
    parser.add_argument(
        "--vocal-vol",
        type=parse_volume,
        default=None,
        help="Vocal guide volume level",
    )
    parser.add_argument(
        "--style",
        choices=["dam", "joysound"],
        default="dam",
        help="Arcade color/layout preset",
    )
    parser.add_argument(
        "--guide-timbre",
        choices=["continuous", "vibraphone", "glockenspiel"],
        default="continuous",
        help="Guide melody synthesis timbre",
    )
    default_whisper = "large-v3"
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb < 6.0:
            default_whisper = "medium"

    parser.add_argument(
        "--whisper-model",
        default=default_whisper,
        help="Whisper model name for stable-ts",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Spoken language for Whisper ('auto', 'ja', 'en', etc.). Default: auto.",
    )
    parser.add_argument(
        "--sep-model",
        default=DEFAULT_SEP_MODEL,
        help="audio-separator model file name",
    )
    parser.add_argument("--font", default=None, help="Subtitle font file")
    parser.add_argument(
        "--force",
        choices=STAGES + [str(i) for i in range(1, 8)],
        default=None,
        help="Regenerate this stage and everything after it",
    )
    args = parser.parse_args()

    vocal_vol = (
        args.vocal_vol
        if args.vocal_vol is not None
        else (1.0 if args.audio_layout == "split-lr" else 0.0)
    )
    for name, val in [
        ("--inst-vol", args.inst_vol),
        ("--guide-vol", args.guide_vol),
        ("--vocal-vol", vocal_vol),
    ]:
        if val > 1.0:
            print(
                f"[warning] {name} gain is {val:.2f} (> 1.0); limiter will prevent clipping."
            )

    verify_ffmpeg()

    song_path = Path(args.input).resolve()
    if not song_path.exists():
        sys.exit(f"Input song not found: {song_path}")

    lyrics_path = Path(args.lyrics).resolve() if args.lyrics else None
    if lyrics_path and not lyrics_path.exists():
        sys.exit(f"Lyrics file not found: {lyrics_path}")

    bg_arg = Path(args.bg).resolve() if args.bg else None
    font_arg = Path(args.font).resolve() if args.font else None

    workspace = resolve_workspace(args.workspace, song_path.stem)
    workspace.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(workspace)

    force_idx = None
    if args.force is not None:
        force_idx = (
            int(args.force) - 1 if args.force.isdigit() else STAGES.index(args.force)
        )
        check_forced_edits(workspace, force_idx, manifest)

    t_pipeline_start = time.perf_counter()
    stage_times: list[tuple[str, float, str]] = []

    def run_stage(idx: int, title: str, fn, skip_reason: str | None = None):
        elapsed_total = time.perf_counter() - t_pipeline_start
        print(f"\n[{idx}/7] {title}  (total {format_duration(elapsed_total)})")
        if skip_reason is not None:
            print(f"  skipped ({skip_reason})")
            stage_times.append((f"[{idx}/7] {title}", 0.0, f"skipped ({skip_reason})"))
            return
        t0 = time.perf_counter()
        forced = force_idx is not None and (idx - 1) >= force_idx
        ran = fn(forced)
        dt = time.perf_counter() - t0
        if ran:
            status = format_duration(dt)
            print(f"  done in {status}")
        else:
            status = "skipped (cached)"
            print(f"  {status}")
        stage_times.append((f"[{idx}/7] {title}", dt, status))

    run_stage(
        1,
        "Decoding input audio",
        lambda f: stage1_decode(song_path, workspace, manifest, f),
    )
    run_stage(
        2,
        "Separating vocal and instrumental stems",
        lambda f: stage2_separate(workspace, args.sep_model, manifest, f),
    )
    run_stage(
        3,
        "Transcribing / aligning lyrics",
        lambda f: stage3_align(
            workspace, lyrics_path, args.whisper_model, args.language, manifest, f
        ),
    )

    font_path = ensure_subtitle_font(font_arg)
    run_stage(
        4,
        "Generating furigana and ASS subtitles",
        lambda f: stage4_ass(workspace, font_path, args.style, manifest, f),
    )
    run_stage(
        5,
        "Extracting and synthesizing guide melody",
        lambda f: stage5_guide(workspace, args.guide_timbre, manifest, f),
        skip_reason=(
            "guide muted"
            if args.guide_vol == 0.0 and force_idx != STAGES.index("guide")
            else None
        ),
    )

    bg_holder: list[Path] = []

    def _run_bg(_forced: bool) -> bool:
        existed = (
            bg_arg is not None or (CACHE_DIR / "bg" / "bg_default_1080p30.mp4").exists()
        )
        bg_holder.append(stage6_background(bg_arg))
        return not existed

    run_stage(6, "Preparing background video", _run_bg)

    out_holder: list[Path] = []

    def _run_render(forced: bool) -> bool:
        out_mp4 = workspace / f"{song_path.stem}_karaoke.mp4"
        old_mtime = out_mp4.stat().st_mtime_ns if out_mp4.exists() else None
        res_path = stage7_render(
            workspace,
            song_path.stem,
            bg_holder[0],
            font_path,
            args.audio_layout,
            args.inst_vol,
            args.guide_vol,
            vocal_vol,
            manifest,
            forced,
        )
        out_holder.append(res_path)
        new_mtime = res_path.stat().st_mtime_ns if res_path.exists() else None
        return old_mtime != new_mtime

    run_stage(7, "Mixing audio and rendering MP4", _run_render)

    total_time = time.perf_counter() - t_pipeline_start
    print("\n--- Pipeline Summary ---")
    for label, _dt, status in stage_times:
        print(f"  {label:<45} {status}")
    print(f"  {'Total':<45} {format_duration(total_time)}")
    print(f"  Output: {out_holder[0]}")


if __name__ == "__main__":
    main()
