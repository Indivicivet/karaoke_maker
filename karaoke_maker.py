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
import librosa
import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, resample_poly, sosfiltfilt
import soundfile as sf
import stable_whisper
import torch
import torch.nn.functional as F
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


def alignment_coverage(line):
    positions = [index for _, index in normalized_characters(line["text"])]
    if not positions:
        return 1.0
    return sum(
        any(chunk["first"] <= index <= chunk["last"] for chunk in line["chunks"])
        for index in positions
    ) / len(positions)


def infer_line_language(text: str, default_lang: str) -> str:
    has_kana = any(0x3040 <= ord(ch) <= 0x30FF for ch in text)
    has_cjk = any(is_kanji(ch) for ch in text)
    if has_kana or (has_cjk and default_lang not in {"zh", "yue"}):
        return "ja"
    if has_cjk:
        return default_lang
    if default_lang in {"ja", "zh", "yue", "ko"} and (
        len(re.findall(r"[A-Za-z']+", text)) >= 2
        or re.search(r"[A-Za-z]{4,}", text) is not None
    ):
        return "en"
    return default_lang


def snap_vocal_boundaries(lines, samples, sample_rate):
    frame_size = max(1, round(sample_rate * 0.02))
    step = frame_size / sample_rate
    frames = samples[: len(samples) // frame_size * frame_size].reshape(-1, frame_size)
    if not len(frames):
        return lines
    rms = np.sqrt(np.mean(frames**2, axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        return lines
    floor = min(peak * 0.08, float(np.percentile(rms, 10)) * 2.5)
    active = rms > max(peak * 0.02, floor)
    onset_floor = max(peak * 0.01, floor * 0.5)
    duration = len(samples) / sample_rate

    copied = [
        {"text": line["text"], "chunks": [dict(c) for c in line["chunks"]]}
        for line in lines
    ]
    flat = [
        (li, ci, chunk)
        for li, line in enumerate(copied)
        for ci, chunk in enumerate(line["chunks"])
    ]
    if not flat:
        return copied

    # Trim silence dragged onto the end of a phrase or line before a rest.
    for idx, (li, ci, chunk) in enumerate(flat):
        next_start = flat[idx + 1][2]["start"] if idx + 1 < len(flat) else duration
        is_line_end = ci == len(copied[li]["chunks"]) - 1
        if not is_line_end and next_start - chunk["end"] < 0.25:
            continue
        f0 = max(0, math.floor(chunk["start"] / step))
        f1 = min(len(active), math.ceil(chunk["end"] / step))
        voiced = np.flatnonzero(active[f0:f1])
        if len(voiced):
            active_end = min(duration, (f0 + int(voiced[-1]) + 1) * step + 0.06)
            if chunk["end"] - active_end > 0.25 and active_end >= chunk["start"] + 0.08:
                chunk["end"] = active_end

    # A quiet lead-in, sustained vocal energy, and non-breath peak distinguish true
    # phrase entrances from intake breaths or brief consonant dips inside a word.
    attacks = []
    for i in np.flatnonzero(np.diff(active.astype(int), prepend=0) == 1):
        if (
            not active[max(0, i - 6) : i].any()
            and len(active[i : i + 4]) == 4
            and active[i : i + 4].all()
            and float(rms[i : min(len(rms), i + 8)].max()) >= peak * 0.05
        ):
            onset = int(i)
            for _ in range(3):
                if (
                    onset > 0
                    and rms[onset - 1] >= onset_floor
                    and rms[onset - 1] <= rms[onset]
                ):
                    onset -= 1
                else:
                    break
            attacks.append(onset * step)

    previous_end = 0.0
    for _, _, chunk in flat:
        if previous_end == 0.0 or chunk["start"] - previous_end >= 0.35:
            nearby = [
                attack
                for attack in attacks
                if previous_end <= attack <= chunk["end"] - 0.06
                and abs(attack - chunk["start"]) <= 0.30
            ]
            if nearby:
                chunk["start"] = min(
                    nearby, key=lambda attack: abs(attack - chunk["start"])
                )
        previous_end = chunk["end"]

    return copied


def recover_untimed_lines(
    model,
    vocals_wav,
    language,
    lines,
    aligned,
    allow_lang_switch: bool = False,
):
    import whisper

    audio = whisper.load_audio(str(vocals_wav))
    audio_dur = len(audio) / 16000.0
    active, frame_seconds = vocal_activity(audio, 16000)

    def has_vocal_near(start: float, end: float) -> bool:
        if not len(active):
            return True
        probe_end = min(end, start + 0.35)
        f0 = max(0, math.floor((start - 0.20) / frame_seconds))
        f1 = min(len(active), math.ceil((probe_end + 0.15) / frame_seconds))
        return bool(active[f0:f1].any())

    def is_bad_line(item: dict, prev_end: float) -> bool:
        alnum_count = sum(1 for _ in normalized_characters(item["text"]))
        if not alnum_count:
            return False
        chunks = item["chunks"]
        if not chunks or alignment_coverage(item) < 1.0:
            return True
        if chunks[0]["start"] + 0.5 < prev_end:
            return True
        if not has_vocal_near(chunks[0]["start"], chunks[-1]["end"]):
            return True
        if alnum_count >= 5 and chunks[-1]["end"] - chunks[0]["start"] < 0.12:
            return True
        return False

    def align_window(slice_lines: list[str], w_start: float, w_end: float, lang: str):
        s0 = max(0, round(w_start * 16000))
        s1 = min(len(audio), round(w_end * 16000))
        if s1 - s0 < 1600:
            return None
        res = model.align(
            audio[s0:s1],
            "\n".join(slice_lines),
            language=lang,
            original_split=True,
            vad=False,
            q_levels=60,
            nonspeech_skip=None,
            fast_mode=True,
        )
        if res is None:
            return None
        mapped = map_alignment_chunks(slice_lines, res.segments)
        offset_sec = s0 / 16000.0
        max_sec = s1 / 16000.0
        for item in mapped:
            valid_chunks = []
            prev_c_end = w_start
            for chunk in item["chunks"]:
                cs = max(prev_c_end, chunk["start"] + offset_sec)
                ce = min(max_sec, chunk["end"] + offset_sec)
                if cs + 1e-3 < ce:
                    valid_chunks.append(
                        {
                            "first": chunk["first"],
                            "last": chunk["last"],
                            "start": cs,
                            "end": ce,
                        }
                    )
                    prev_c_end = ce
            item["chunks"] = valid_chunks
        return mapped

    recovered = [
        {"text": item["text"], "chunks": [dict(c) for c in item["chunks"]]}
        for item in aligned
    ]
    line_langs = [
        infer_line_language(line, language) if allow_lang_switch else language
        for line in lines
    ]

    # Re-align foreign-script blocks (e.g. English choruses/bridges inside a Japanese
    # song) using their native tokenizer and cross-attention language token.
    if allow_lang_switch:
        idx = 0
        while idx < len(lines):
            if line_langs[idx] == language:
                idx += 1
                continue
            block_lang = line_langs[idx]
            end_idx = idx + 1
            while end_idx < len(lines) and line_langs[end_idx] == block_lang:
                end_idx += 1
            lower = max(
                (c["end"] for item in recovered[:idx] for c in item["chunks"]),
                default=0.0,
            )
            upper = min(
                (c["start"] for item in recovered[end_idx:] for c in item["chunks"]),
                default=audio_dur,
            )
            block_items = recovered[idx:end_idx]
            k = end_idx - idx
            all_timed = all(item["chunks"] for item in block_items)
            w_start, w_end = lower, upper
            if all_timed:
                cur_s = block_items[0]["chunks"][0]["start"]
                cur_e = max(c["end"] for item in block_items for c in item["chunks"])
                if (
                    lower <= cur_s + 0.25
                    and cur_e <= upper + 0.25
                    and has_vocal_near(cur_s, cur_e)
                ):
                    w_start = max(lower, cur_s - 0.5)
                    w_end = min(upper, cur_e + 0.5)
            if w_end > w_start + 0.15 * k and (
                all_timed or (w_end - w_start) <= 8.0 * k + 6.0
            ):
                cand_block = align_window(
                    lines[idx:end_idx], w_start, w_end, block_lang
                )
                if cand_block is not None:
                    old_cov = sum(alignment_coverage(it) for it in block_items) / k
                    new_cov = sum(alignment_coverage(it) for it in cand_block) / k
                    if new_cov >= old_cov and (
                        not all_timed or all(it["chunks"] for it in cand_block)
                    ):
                        recovered[idx:end_idx] = cand_block
                        print(
                            f"  Aligned lines {idx + 1}-{end_idx} as '{block_lang}' "
                            f"({w_start:.1f}-{w_end:.1f}s)."
                        )
            idx = end_idx

    def any_needs_recovery() -> bool:
        prev_e = 0.0
        for item in recovered:
            if is_bad_line(item, prev_e):
                return True
            if item["chunks"]:
                prev_e = max(prev_e, item["chunks"][-1]["end"])
        return False

    if not any_needs_recovery():
        return recovered

    # First try bounded neighbor windows for poorly timed blocks before falling back
    # to full-song transcription text matching.
    idx = 0
    while idx < len(lines):
        prev_end = max(
            (c["end"] for item in recovered[:idx] for c in item["chunks"]),
            default=0.0,
        )
        if not is_bad_line(recovered[idx], prev_end):
            idx += 1
            continue
        block_lang = line_langs[idx]
        end_idx = idx + 1
        probe_prev = (
            max(prev_end, recovered[idx]["chunks"][-1]["end"])
            if recovered[idx]["chunks"]
            else prev_end
        )
        while (
            end_idx < len(lines)
            and line_langs[end_idx] == block_lang
            and is_bad_line(recovered[end_idx], probe_prev)
        ):
            if recovered[end_idx]["chunks"]:
                probe_prev = max(probe_prev, recovered[end_idx]["chunks"][-1]["end"])
            end_idx += 1

        next_anchor = None
        for g in range(end_idx, len(lines)):
            g_chunks = recovered[g]["chunks"]
            if (
                g_chunks
                and alignment_coverage(recovered[g]) == 1.0
                and g_chunks[0]["start"] > prev_end + 0.1
                and has_vocal_near(g_chunks[0]["start"], g_chunks[-1]["end"])
            ):
                next_anchor = g_chunks[0]["start"]
                break
        upper = next_anchor if next_anchor is not None else audio_dur
        k = end_idx - idx
        bounded = (
            idx > 0 and next_anchor is not None and (upper - prev_end) <= 8.0 * k + 6.0
        ) or ((upper - prev_end) <= 5.0 * k + 2.0)
        if upper > prev_end + 0.15 * k and bounded:
            cand_block = align_window(lines[idx:end_idx], prev_end, upper, block_lang)
            if cand_block is not None:
                cur_prev = prev_end
                for offset, cand in enumerate(cand_block, start=idx):
                    old_item = recovered[offset]
                    if cand["chunks"] and cand["chunks"][0]["start"] >= cur_prev - 1e-3:
                        if alignment_coverage(cand) > alignment_coverage(old_item) or (
                            alignment_coverage(cand) >= alignment_coverage(old_item)
                            and not is_bad_line(cand, cur_prev)
                        ):
                            recovered[offset] = cand
                            print(
                                f"  Re-aligned lyric line {offset + 1} in bounded "
                                f"window {prev_end:.1f}-{upper:.1f}s."
                            )
                    if recovered[offset]["chunks"]:
                        cur_prev = max(cur_prev, recovered[offset]["chunks"][-1]["end"])
        idx = end_idx

    def needs_transcript_search(
        item: dict, prev_end: float, upper_bound: float
    ) -> bool:
        if upper_bound - prev_end <= 6.0:
            return False
        alnum_count = sum(1 for _ in normalized_characters(item["text"]))
        if not alnum_count:
            return False
        chunks = item["chunks"]
        if not chunks or alignment_coverage(item) < 0.65:
            return True
        if chunks[0]["start"] + 0.5 < prev_end:
            return True
        return not has_vocal_near(chunks[0]["start"], chunks[-1]["end"])

    transcript_slices = {}
    announced_search = False
    for index, line in enumerate(lines):
        previous_end = max(
            (chunk["end"] for item in recovered[:index] for chunk in item["chunks"]),
            default=0.0,
        )
        upper_bound = audio_dur
        for g in range(index + 1, len(lines)):
            g_chunks = recovered[g]["chunks"]
            if (
                g_chunks
                and alignment_coverage(recovered[g]) == 1.0
                and g_chunks[0]["start"] > previous_end + 0.1
                and has_vocal_near(g_chunks[0]["start"], g_chunks[-1]["end"])
            ):
                upper_bound = g_chunks[0]["start"]
                break

        if not needs_transcript_search(recovered[index], previous_end, upper_bound):
            continue
        expected = "".join(char for char, _ in normalized_characters(line))
        if not expected:
            continue
        if not announced_search:
            print("  Searching vocal transcription windows for unlocated lyric lines.")
            announced_search = True

        line_lang = line_langs[index]
        gap_s = max(0.0, previous_end - 0.5)
        gap_e = min(audio_dur, upper_bound + 1.5)
        cache_key = (line_lang, round(gap_s, 1), round(gap_e, 1))
        if cache_key not in transcript_slices:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sub_audio = audio[round(gap_s * 16000) : round(gap_e * 16000)]
            sub_res = model.transcribe(
                sub_audio,
                language=line_lang,
                vad=False,
                stream=False,
                fp16=False,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            transcript_slices[cache_key] = [
                (seg.start + gap_s, seg.end + gap_s, seg.text)
                for seg in sub_res.segments
            ]
        seg_list = transcript_slices[cache_key]

        candidates = []
        for position, (seg_s, seg_e, seg_text) in enumerate(seg_list):
            if seg_e <= previous_end or seg_s >= upper_bound + 1.0:
                continue
            heard = seg_text
            end_s = seg_e
            for next_s, next_e, next_text in seg_list[position : position + 2]:
                if (next_s, next_e, next_text) != (seg_s, seg_e, seg_text):
                    if next_s - seg_e > 4 or next_s >= upper_bound + 1.0:
                        break
                    heard += next_text
                    end_s = next_e
                normalized = "".join(char for char, _ in normalized_characters(heard))
                match = difflib.SequenceMatcher(
                    None, expected, normalized, autojunk=False
                )
                score = sum(block.size for block in match.get_matching_blocks()) / len(
                    expected
                )
                if score >= 0.65:
                    candidates.append((score, seg_s, end_s))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if recovered[index]["chunks"]:
            original_start = recovered[index]["chunks"][0]["start"]
            nearby = [
                candidate
                for candidate in candidates
                if abs(candidate[1] - original_start) <= 15
            ]
            if nearby:
                candidates = nearby
        alternatives = (
            [
                candidate
                for candidate in candidates[1:]
                if candidate[1] >= candidates[0][2] or candidate[2] <= candidates[0][1]
            ]
            if candidates
            else []
        )
        if alternatives and recovered[index]["chunks"]:
            original_start = recovered[index]["chunks"][0]["start"]
            closest = sorted(
                [candidates[0], *alternatives],
                key=lambda candidate: abs(candidate[1] - original_start),
            )
            if (
                abs(closest[0][1] - original_start) <= 15
                and abs(closest[1][1] - original_start)
                - abs(closest[0][1] - original_start)
                >= 15
            ):
                candidates = [closest[0]]
                alternatives = []
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
        end = min(upper_bound, candidates[0][2] + 1.0)
        if end <= start + 0.1:
            end = min(audio_dur, candidates[0][2] + 1.0)
        cand_list = align_window([line], start, end, line_lang)
        if not cand_list:
            continue
        candidate = cand_list[0]
        if candidate["chunks"] and (
            alignment_coverage(candidate) > alignment_coverage(recovered[index])
            or (
                alignment_coverage(candidate) >= alignment_coverage(recovered[index])
                and is_bad_line(recovered[index], previous_end)
                and not is_bad_line(candidate, previous_end)
            )
        ):
            recovered[index] = candidate
            print(f"  Re-aligned lyric line {index + 1} at {start:.1f}s.")

    # Ensure no lyric line with sung characters is left completely empty and dropped
    # from karaoke.ass when a valid time gap exists between neighboring timed lines.
    idx = 0
    while idx < len(recovered):
        positions = [pos for _, pos in normalized_characters(recovered[idx]["text"])]
        if recovered[idx]["chunks"] or not positions:
            idx += 1
            continue
        end_idx = idx + 1
        while (
            end_idx < len(recovered)
            and not recovered[end_idx]["chunks"]
            and any(normalized_characters(recovered[end_idx]["text"]))
        ):
            end_idx += 1
        lower = max(
            (c["end"] for item in recovered[:idx] for c in item["chunks"]),
            default=0.0,
        )
        upper = min(
            (c["start"] for item in recovered[end_idx:] for c in item["chunks"]),
            default=audio_dur,
        )
        count = end_idx - idx
        if upper > lower + 0.05 * count:
            t0, t1 = lower, upper
            if len(active):
                f0 = max(0, math.floor(lower / frame_seconds))
                f1 = min(len(active), math.ceil(upper / frame_seconds))
                voiced = np.flatnonzero(active[f0:f1])
                if len(voiced):
                    v0 = max(lower, (f0 + int(voiced[0])) * frame_seconds)
                    v1 = min(upper, (f0 + int(voiced[-1]) + 1) * frame_seconds)
                    if v1 - v0 >= 0.12 * count:
                        t0, t1 = v0, v1
            char_spans = [
                [pos for _, pos in normalized_characters(recovered[i]["text"])]
                for i in range(idx, end_idx)
            ]
            weights = [max(1, len(sp)) for sp in char_spans]
            total_w = sum(weights)
            cursor = t0
            for offset, (sp, w) in enumerate(zip(char_spans, weights), start=idx):
                dur = (t1 - t0) * w / total_w
                recovered[offset] = {
                    "text": recovered[offset]["text"],
                    "chunks": [
                        {
                            "first": sp[0],
                            "last": sp[-1],
                            "start": cursor,
                            "end": cursor + dur,
                        }
                    ],
                }
                cursor += dur
                print(
                    f"  Interpolated fallback timing for lyric line {offset + 1} "
                    f"({recovered[offset]['chunks'][0]['start']:.2f}-"
                    f"{recovered[offset]['chunks'][0]['end']:.2f}s)."
                )
        idx = end_idx

    return recovered


def vocal_activity(samples, sample_rate):
    frame_size = max(1, round(sample_rate * 0.02))
    frame_seconds = frame_size / sample_rate
    frames = samples[: len(samples) // frame_size * frame_size].reshape(-1, frame_size)
    if not len(frames):
        return np.empty(0, dtype=bool), frame_seconds
    rms = np.sqrt(np.mean(frames**2, axis=1))
    peak = float(rms.max())
    # Dense vocal tracks can have singing across >80% of frames, so use p10 rather
    # than p20 to estimate background noise without mistaking quiet verses for silence.
    floor = min(peak * 0.08, float(np.percentile(rms, 10)) * 2.5)
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
            "version": 8,
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

    explicit_lang = (
        language.lower() if (language and language.lower() != "auto") else None
    )

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

        if explicit_lang is not None:
            resolved_lang = explicit_lang
        elif any(0x3040 <= ord(ch) <= 0x30FF for line in clean_lines for ch in line):
            resolved_lang = "ja"
            print("  Detected Japanese script in lyrics; aligning primarily as 'ja'.")
        else:
            resolved_lang = detect_audio_language(model, vocals_wav)
            if resolved_lang in {"ja", "zh", "yue", "ko"} and not any(
                ord(ch) > 0x024F and ch.isalpha() for line in clean_lines for ch in line
            ):
                print(
                    f"  Lyrics use Latin script; overriding audio language "
                    f"'{resolved_lang}' with 'en'."
                )
                resolved_lang = "en"

        # Speech VAD misclassifies sustained, belted, or screamed vocals as silence;
        # default q_levels=20 clips soft singing (~-27 dB), and default word_dur_factor=2.0
        # treats held sung notes (>2x median word length) as misalignments.
        result = model.align(
            str(vocals_wav),
            joined_text,
            language=resolved_lang,
            original_split=True,
            vad=False,
            q_levels=60,
            word_dur_factor=3.5,
            max_word_dur=4.5,
        )
        if result is None:
            raise ValueError(
                "Whisper could not align the supplied lyrics to vocals.wav."
            )
    else:
        resolved_lang = (
            explicit_lang
            if explicit_lang is not None
            else detect_audio_language(model, vocals_wav)
        )
        transcribe_kwargs = {
            "language": resolved_lang,
            "stream": False,
            "condition_on_previous_text": False,
        }
        if device == "cuda" and total_vram_gb < 6.0:
            transcribe_kwargs["fp16"] = False
            transcribe_kwargs["temperature"] = 0.0
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

    samples, rate = sf.read(vocals_wav, dtype="float32", always_2d=True)
    mono_vocals = samples.mean(axis=1)
    out_lines = snap_vocal_boundaries(
        map_alignment_chunks(clean_lines, result.segments), mono_vocals, rate
    )
    if has_user_lyrics:
        out_lines = recover_untimed_lines(
            model,
            vocals_wav,
            resolved_lang,
            clean_lines,
            out_lines,
            allow_lang_switch=(explicit_lang is None),
        )
        out_lines = snap_vocal_boundaries(out_lines, mono_vocals, rate)
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

    if not chunks:
        return [(0.0, 0.0)] * n

    # Share adjacent chunk intervals when leading, trailing, or zero-gap middle
    # words lacked distinct timestamps, avoiding 1 ms visual flashes.
    spans = [[c["first"], c["last"] + 1, c["start"], c["end"]] for c in chunks]
    if sum(char_mora_weights[0 : spans[0][0]]) > 0:
        spans[0][0] = 0
    if sum(char_mora_weights[spans[-1][1] : n]) > 0:
        spans[-1][1] = n

    merged_spans = [spans[0]]
    for sp in spans[1:]:
        prev = merged_spans[-1]
        gap_weight = sum(char_mora_weights[prev[1] : sp[0]])
        if gap_weight > 0 and sp[2] - prev[3] < 0.02:
            prev[1] = sp[1]
            prev[3] = sp[3]
        else:
            merged_spans.append(sp)

    next_char = 0
    previous_end = merged_spans[0][2]
    for first_c, stop_c, start_t, end_t in merged_spans:
        distribute(next_char, first_c, previous_end, start_t)
        next_char = stop_c
        distribute(first_c, next_char, start_t, end_t)
        previous_end = end_t
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
        {"stage": "4_ass", "version": 3, "style": style_preset},
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

    audio_dur = len(samples) / sample_rate
    effective_chunks = [list(alignment["chunks"]) for alignment in align_data]
    idx = 0
    while idx < len(effective_chunks):
        if effective_chunks[idx] or not display_lines[idx][0].strip():
            idx += 1
            continue
        end_idx = idx + 1
        while (
            end_idx < len(effective_chunks)
            and not effective_chunks[end_idx]
            and display_lines[end_idx][0].strip()
        ):
            end_idx += 1
        lower = max(
            (c["end"] for chs in effective_chunks[:idx] for c in chs), default=0.0
        )
        upper = min(
            (c["start"] for chs in effective_chunks[end_idx:] for c in chs),
            default=audio_dur,
        )
        count = end_idx - idx
        if upper > lower + 0.02 * count:
            step_dur = (upper - lower) / count
            for offset in range(idx, end_idx):
                plain_line = display_lines[offset][0]
                effective_chunks[offset] = [
                    {
                        "first": 0,
                        "last": len(plain_line) - 1,
                        "start": lower + (offset - idx) * step_dur,
                        "end": lower + (offset - idx + 1) * step_dur,
                    }
                ]
        idx = end_idx

    parsed_lines = []
    for (plain_text, segments), chunks in zip(display_lines, effective_chunks):
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
        active_rms = rms[rms > float(np.max(rms)) * 0.02]
        reference = (
            max(float(np.percentile(active_rms, 90)), 1e-6)
            if len(active_rms)
            else max(float(np.percentile(rms, 95)), 1e-6)
        )
        # Compress vocal dynamics so quiet verses remain clearly audible under rock mixes.
        envelope = np.clip(rms / reference, 0.0, 1.0) ** 0.42
        frame_times = (np.arange(len(rms)) + 0.5) * hop / vocal_rate

    notes = read_guide_notes(midi_path, duration_sec)
    partials = [(1.0, 1.0, 0.0), (2.0, 0.28, 0.0), (3.0, 0.09, 12.0), (4.0, 0.04, 18.0)]
    for idx, (start, end, pitch) in enumerate(notes):
        first = max(0, round(start * sr))
        stop = min(len(buf), round(end * sr))
        length = stop - first
        if length <= 0:
            continue
        t = np.arange(length, dtype=np.float32) / float(sr)
        f0 = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))
        is_legato = idx > 0 and (start - notes[idx - 1][1]) <= 0.025
        # Gentle vibrato after 200 ms on held notes, classic DAM guide synth style.
        vib_env = np.clip((t - 0.20) / 0.14, 0.0, 1.0)
        phase = 2.0 * np.pi * f0 * t + (0.0028 * f0) * vib_env * np.sin(
            2.0 * np.pi * 5.2 * t
        )
        tone = np.zeros(length, dtype=np.float32)
        chiff_scale = 0.45 if is_legato else 1.0
        for ratio, amp, decay in partials:
            if f0 * ratio < sr * 0.45:
                env_p = np.exp(-decay * t) * chiff_scale if decay > 0 else 1.0
                tone += amp * env_p * np.sin(ratio * phase)

        if len(rms):
            tone *= 0.12 + 0.88 * np.interp(
                np.arange(first, stop) / sr, frame_times, envelope
            )
        att_len = min(round((0.016 if is_legato else 0.010) * sr), length // 2)
        rel_len = min(round(0.016 * sr), length // 2)
        if att_len:
            tone[:att_len] *= np.sin(np.linspace(0, np.pi / 2, att_len)) ** 2
        if rel_len:
            tone[-rel_len:] *= np.sin(np.linspace(0, np.pi / 2, rel_len))[::-1] ** 2
        buf[first:stop] += tone

    peak = float(np.max(np.abs(buf), initial=0.0))
    if peak > 1e-6:
        # Soft-clip rare note overlaps so one transient cannot pull the whole guide down.
        scale_ref = max(
            (
                float(np.percentile(np.abs(buf[buf != 0]), 99.5))
                if np.any(buf != 0)
                else peak
            ),
            1e-6,
        )
        buf = np.tanh(buf / (scale_ref * 1.15)) * (10 ** (-1 / 20))
    sf.write(str(out_wav), np.column_stack([buf, buf]), sr, subtype="FLOAT")


def synthesize_filtered_vocals(vocals_wav: Path, out_wav: Path):
    """Create a softened, de-sung shadow vocal directly from vocals.wav without lyrics/Whisper."""
    audio, sr = sf.read(str(vocals_wav), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    n_samples = len(mono)

    if n_samples < 32:
        sf.write(str(out_wav), audio, sr, subtype="FLOAT")
        return

    sos_hp = butter(2, 100.0, btype="highpass", fs=sr, output="sos")
    sig = sosfiltfilt(sos_hp, mono).astype(np.float32)

    n_fft = 2048
    hop = 512
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    pad_len = n_fft
    sig_pad = np.pad(sig, (pad_len, pad_len))
    frames = np.lib.stride_tricks.sliding_window_view(sig_pad, n_fft)[::hop]
    n_frames = len(frames)
    spec = np.fft.rfft(frames * win, axis=1)
    mag = np.abs(spec)
    phase = np.angle(spec)

    # Vectorized HPSS median filter (on GPU when available) to isolate pitched harmonics
    # while softly attenuating consonants, sibilance, breaths, and scream grit.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mag_t = torch.from_numpy(mag).to(device)
    h_med = (
        F.pad(mag_t.T.unsqueeze(0), (7, 7), mode="reflect")
        .squeeze(0)
        .unfold(1, 15, 1)
        .median(dim=-1)
        .values.T
    )
    p_med = (
        F.pad(mag_t.unsqueeze(0), (8, 8), mode="reflect")
        .squeeze(0)
        .unfold(1, 17, 1)
        .median(dim=-1)
        .values
    )
    tonal_mask = (
        ((h_med * h_med) / (h_med * h_med + p_med * p_med + 1e-12)).cpu().numpy()
    )
    del mag_t, h_med, p_med
    if device == "cuda":
        torch.cuda.empty_cache()

    mag_tonal = mag * (0.15 + 0.85 * tonal_mask)

    # Cepstral vocal-tract smoothing: dampen low-quefrency vowel formants (0.1..1.8 ms)
    # while preserving the higher-quefrency harmonic pitch comb.
    log_mag = np.log(mag_tonal + 1e-6)
    ceps = np.fft.irfft(log_mag, n=n_fft, axis=1)
    q_low = max(3, round(0.0001 * sr))
    q_high = round(0.0018 * sr)
    ceps[:, q_low:q_high] *= 0.35
    ceps[:, n_fft - q_high + 1 : n_fft - q_low + 1] *= 0.35
    mag_desung = np.exp(np.fft.rfft(ceps, n=n_fft, axis=1).real)
    mag_blend = 0.70 * mag_desung + 0.30 * mag_tonal

    # Gentle warm rolloff above 1.7 kHz and mild 2.6 kHz presence dip.
    lpf = 1.0 / np.sqrt(1.0 + (freqs / 1700.0) ** 4)
    presence_dip = 1.0 - 0.35 * np.exp(-0.5 * ((freqs - 2600.0) / 700.0) ** 2)
    mag_eq = mag_blend * (lpf * presence_dip).astype(np.float32)[None, :]

    recon_frames = np.fft.irfft(mag_eq * np.exp(1j * phase), n=n_fft, axis=1) * win
    out_pad = np.zeros(len(sig_pad), dtype=np.float32)
    win_norm = np.zeros(len(sig_pad), dtype=np.float32)
    win_sq = win * win
    for idx in range(n_frames):
        s = idx * hop
        out_pad[s : s + n_fft] += recon_frames[idx]
        win_norm[s : s + n_fft] += win_sq
    out_mono = out_pad[pad_len : pad_len + n_samples] / np.maximum(
        win_norm[pad_len : pad_len + n_samples], 1e-6
    )

    # Smooth dynamic leveling so quiet verses remain audible and loud screams stay tame.
    env_win = max(1, round(0.04 * sr))
    rms_env = np.sqrt(
        gaussian_filter1d(out_mono * out_mono, sigma=env_win) + 1e-12
    ).astype(np.float32)
    active_env = rms_env[rms_env > float(np.percentile(rms_env, 20)) * 2.0]
    target_rms = (
        float(np.median(active_env)) if len(active_env) else float(np.max(rms_env))
    )
    if target_rms > 1e-6:
        floor = target_rms * 0.08
        gain = np.where(
            rms_env > floor,
            (target_rms / np.maximum(rms_env, floor)) ** 0.45,
            rms_env / max(floor, 1e-6),
        )
        out_mono = out_mono * np.clip(gain, 0.25, 2.5)

    # Subtle stereo diffusion so the softened vocal sits behind the singer.
    d1 = round(0.011 * sr)
    d2 = round(0.017 * sr)
    left = out_mono.copy()
    right = out_mono.copy()
    if n_samples > d2:
        left[d1:] += 0.18 * out_mono[:-d1]
        right[d2:] += 0.18 * out_mono[:-d2]
        left[d2:] -= 0.10 * out_mono[:-d2]
        right[d1:] -= 0.10 * out_mono[:-d1]
    stereo = np.column_stack([left, right])

    peak = float(np.max(np.abs(stereo), initial=0.0))
    if peak > 1e-6:
        stereo *= (10.0 ** (-1.0 / 20.0)) / peak
    sf.write(str(out_wav), stereo, sr, subtype="FLOAT")


def decode_guide_pitch(probabilities):
    """Viterbi decode CREPE sigmoid activations while allowing melodic leaps."""
    if not hasattr(decode_guide_pitch, "_transition"):
        idx = np.arange(360)
        dist = np.abs(idx[:, None] - idx[None, :])
        local = np.maximum(12.0 - dist, 0.0)
        trans = local + 0.004 * np.exp(-dist / 7.0) + 1e-6
        trans /= trans.sum(axis=1, keepdims=True)
        decode_guide_pitch._transition = trans

    probs_np = np.clip(probabilities.detach().cpu().numpy(), 1e-6, 1.0)
    probs_np /= probs_np.sum(axis=1, keepdims=True)
    bins_np = np.array(
        [
            librosa.sequence.viterbi(seq, decode_guide_pitch._transition).astype(
                np.int64
            )
            for seq in probs_np
        ]
    )
    bins = torch.tensor(bins_np, device=probabilities.device)
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

    def finish_note(stop, min_edge_frames: int = 6):
        pitch = round(float(np.median(pitches[pitch_start:stop])))
        if stop - pitch_start >= min_edge_frames or stable_short_note(
            pitch_start, stop, pitch
        ):
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
                    finish_note(end)
                    start = end
                    pitch_start = end
        end += 1

    if end > pitch_start:
        finish_note(end)
    return notes


def clean_guide_notes(
    notes: list[tuple[float, float, int]],
    audio_16k: np.ndarray,
    sr: int,
    lyric_starts: list[float],
    chunk_starts: list[float],
) -> list[tuple[float, float, int]]:
    """Remove brief harmonic spikes, resolve octaves, and re-articulate repeated lyric chunks."""
    if not notes:
        return notes

    cleaned = list(notes)
    for _ in range(3):
        out = []
        i = 0
        changed = False
        while i < len(cleaned):
            if out and i + 1 < len(cleaned):
                t0_a, t1_a, p_a = out[-1]
                t0_b, t1_b, p_b = cleaned[i]
                t0_c, t1_c, p_c = cleaned[i + 1]
                if (
                    (t1_b - t0_b) < 0.25
                    and (t0_b - t1_a) <= 0.12
                    and (t0_c - t1_b) <= 0.12
                    and abs(p_b - p_a) >= 7
                    and abs(p_b - p_c) >= 7
                    and abs(p_a - p_c) <= 3
                ):
                    if p_a == p_c and (t0_c - t1_a) <= 0.30:
                        out[-1] = (t0_a, t1_c, p_a)
                        i += 2
                    else:
                        i += 1
                    changed = True
                    continue
            out.append(cleaned[i])
            i += 1
        cleaned = out
        if not changed:
            break

    if lyric_starts:
        cleaned = correct_guide_octaves(cleaned, audio_16k, sr, lyric_starts)

    # Also resolve isolated octave excursions before/between lyric phrases.
    resolved = []
    for i, (t0_b, t1_b, p_b) in enumerate(cleaned):
        if resolved and i + 1 < len(cleaned):
            _, t1_a, p_a = resolved[-1]
            t0_c, _, p_c = cleaned[i + 1]
            delta = p_b - p_a
            if (
                abs(delta) in (11, 12, 13)
                and abs(p_c - p_a) <= 2
                and (t0_b - t1_a) <= 0.25
                and (t0_c - t1_b) <= 0.25
                and (t1_b - t0_b) >= 0.10
            ):
                alt = p_b - 12 if delta > 0 else p_b + 12
                ratio = octave_harmonic_ratio(audio_16k, sr, t0_b, t1_b, min(p_b, alt))
                if ratio is not None and (
                    (delta < 0 and ratio < 0.45) or (delta > 0 and ratio > 1.1)
                ):
                    resolved.append((t0_b, t1_b, alt))
                    continue
        resolved.append((t0_b, t1_b, p_b))

    if not chunk_starts:
        return resolved

    # Re-articulate long sustained notes at distinct lyric chunk onsets.
    articulated = []
    for t0, t1, pitch in resolved:
        if t1 - t0 < 0.30:
            articulated.append((t0, t1, pitch))
            continue
        splits = [cs for cs in chunk_starts if t0 + 0.12 <= cs <= t1 - 0.10]
        cur_start = t0
        for cs in splits:
            if cs - cur_start >= 0.12 and t1 - cs >= 0.10:
                articulated.append((cur_start, cs - 0.012, pitch))
                cur_start = cs
        articulated.append((cur_start, t1, pitch))
    return articulated


def stage5_guide(
    workspace: Path,
    guide_timbre: str,
    manifest: dict,
    force: bool,
    guide_vol: float = 0.3,
    filtered_vocal_vol: float = 0.0,
) -> bool:
    source_wav = workspace / "source.wav"
    vocals_wav = workspace / "vocals.wav"
    alignment_json = workspace / "alignment.json"
    guide_mid = workspace / "guide.mid"
    guide_wav = workspace / "guide.wav"
    filtered_wav = workspace / "vocals_filtered.wav"

    need_guide = guide_vol > 0.0 or force
    need_filtered = filtered_vocal_vol > 0.0 or force
    did_work = False

    if need_guide:
        mid_fp = compute_fingerprint(
            [vocals_wav, alignment_json],
            {"stage": "5_mid_v7", "timbre": guide_timbre},
        )
        run_mid, kept_mid = check_stage_status([guide_mid], mid_fp, manifest, force)

        if run_mid:
            audio, sr = sf.read(str(vocals_wav), dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            target_sr = 16000
            if sr != target_sr:
                g = math.gcd(sr, target_sr)
                mono_16k = resample_poly(mono, target_sr // g, sr // g).astype(
                    np.float32
                )
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

            active_rms = rms[rms > 1e-4]
            ref_rms = float(np.percentile(active_rms, 95)) if len(active_rms) else 0.0
            rms_gate = max(
                float(np.max(rms)) * 0.025 if len(rms) else 0.0, ref_rms * 0.04
            )

            align_data = json.loads(alignment_json.read_text(encoding="utf-8")).get(
                "lines", []
            )
            chunks = [chunk for line in align_data for chunk in line.get("chunks", [])]
            strong_voiced = (pd_np >= 0.35) & (rms > rms_gate) & (f0_np >= 45.0)
            if chunks:
                lyric_mask = np.zeros(n_frames, dtype=bool)
                for chunk in chunks:
                    start = max(0, math.floor((chunk["start"] - 0.15) / 0.01))
                    stop = min(n_frames, math.ceil((chunk["end"] + 0.15) / 0.01))
                    lyric_mask[start:stop] = True
                voiced = strong_voiced | (
                    (pd_np >= 0.20)
                    & (rms > rms_gate * 0.85)
                    & (f0_np >= 45.0)
                    & lyric_mask
                )
            else:
                voiced = strong_voiced

            valid_f0 = np.maximum(f0_np, 1e-3)
            raw_midi = 69.0 + 12.0 * np.log2(valid_f0 / 440.0)

            # Reconnect held pitches when CREPE loses confidence through a consonant.
            voiced_bridged = voiced.copy()
            midi_bridged = raw_midi.copy()
            idx = 0
            while idx < n_frames:
                if not voiced_bridged[idx]:
                    end_idx = idx
                    while end_idx < n_frames and not voiced_bridged[end_idx]:
                        end_idx += 1
                    gap_len = end_idx - idx
                    if idx >= 2 and end_idx + 2 <= n_frames and gap_len <= 12:
                        before = slice(idx - 2, idx)
                        after = slice(end_idx, end_idx + 2)
                        p_prev = float(np.median(raw_midi[before]))
                        p_next = float(np.median(raw_midi[after]))
                        min_energy = 0.20 * min(
                            np.median(rms[before]), np.median(rms[after])
                        )
                        if (
                            voiced[before].all()
                            and voiced[after].all()
                            and abs(p_next - p_prev) <= 1.0
                            and np.min(rms[idx:end_idx])
                            > max(rms_gate * 0.6, min_energy)
                        ):
                            voiced_bridged[idx:end_idx] = True
                            midi_bridged[idx:end_idx] = np.linspace(
                                p_prev, p_next, gap_len + 2
                            )[1:-1]
                    idx = end_idx
                else:
                    idx += 1

            midi_filt = np.zeros(n_frames, dtype=np.float32)
            idx = 0
            while idx < n_frames:
                if voiced_bridged[idx]:
                    end_idx = idx
                    while end_idx < n_frames and voiced_bridged[end_idx]:
                        end_idx += 1
                    seg = midi_bridged[idx:end_idx]
                    if len(seg) >= 5:
                        midi_filt[idx:end_idx] = median_filter(
                            seg, size=5, mode="nearest"
                        )
                    elif len(seg) >= 3:
                        midi_filt[idx:end_idx] = median_filter(
                            seg, size=3, mode="nearest"
                        )
                    else:
                        midi_filt[idx:end_idx] = seg
                    idx = end_idx
                else:
                    idx += 1

            raw_notes = guide_pitch_notes(midi_filt, pd_np)
            lyric_starts = sorted(
                min(chunk["start"] for chunk in line["chunks"])
                for line in align_data
                if line.get("chunks")
            )
            chunk_starts = sorted(chunk["start"] for chunk in chunks)
            notes = clean_guide_notes(
                raw_notes, mono_16k, target_sr, lyric_starts, chunk_starts
            )

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
                    Message(
                        "note_on", note=pitch, velocity=100, time=on_tick - last_tick
                    )
                )
                track.append(
                    Message("note_off", note=pitch, velocity=0, time=off_tick - on_tick)
                )
                last_tick = off_tick

            mid.save(str(guide_mid))
            record_artifacts([guide_mid], kept_mid, mid_fp, manifest, workspace)
            did_work = True

        wav_inputs = [guide_mid, source_wav]
        if guide_timbre == "continuous":
            wav_inputs.append(vocals_wav)
        wav_fp = compute_fingerprint(
            wav_inputs, {"stage": "5_wav_v6", "timbre": guide_timbre}
        )
        run_wav, kept_wav = check_stage_status([guide_wav], wav_fp, manifest, force)
        if run_wav:
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
                    guide_mid,
                    guide_wav,
                    info.duration,
                    guide_timbre,
                    sr=info.samplerate,
                )
            record_artifacts([guide_wav], kept_wav, wav_fp, manifest, workspace)
            did_work = True

    if need_filtered:
        filt_fp = compute_fingerprint([vocals_wav], {"stage": "5_filtered_v1"})
        run_filt, kept_filt = check_stage_status(
            [filtered_wav], filt_fp, manifest, force
        )
        if run_filt:
            synthesize_filtered_vocals(vocals_wav, filtered_wav)
            record_artifacts([filtered_wav], kept_filt, filt_fp, manifest, workspace)
            did_work = True

    return did_work


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
    filtered_vocal_vol: float = 0.0,
) -> Path:
    source_wav = workspace / "source.wav"
    inst_wav = workspace / "instrumental.wav"
    guide_wav = workspace / "guide.wav"
    filtered_wav = workspace / "vocals_filtered.wav"
    vocals_wav = workspace / "vocals.wav"
    karaoke_ass = workspace / "karaoke.ass"
    output_mp4 = workspace / f"{song_stem}_karaoke.mp4"

    use_guide = guide_vol > 0.0
    use_filtered = filtered_vocal_vol > 0.0
    upstream_files = [inst_wav]
    if use_guide:
        upstream_files.append(guide_wav)
    if use_filtered:
        upstream_files.append(filtered_wav)
    upstream_files.extend([vocals_wav, karaoke_ass, bg_path, font_path])

    input_fp = compute_fingerprint(
        upstream_files,
        {
            "stage": 7,
            "audio_layout": audio_layout,
            "inst_vol": inst_vol,
            "guide_vol": guide_vol,
            "filtered_vocal_vol": filtered_vocal_vol,
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

    audio_stems = [("inst", inst_wav, inst_vol)]
    if use_guide:
        audio_stems.append(("guide", guide_wav, guide_vol))
    if use_filtered:
        audio_stems.append(("filt", filtered_wav, filtered_vocal_vol))
    audio_stems.append(("voc", vocals_wav, vocal_vol))

    audio_inputs = []
    for _, wav_path, _ in audio_stems:
        audio_inputs.extend(["-i", str(wav_path)])

    if audio_layout == "mix":
        gain_filters = [
            f"[{idx}:a]volume={gain}[a_{name}]"
            for idx, (name, _, gain) in enumerate(audio_stems, start=1)
        ]
        mix_labels = "".join(f"[a_{name}]" for name, _, _ in audio_stems)
        filter_complex = (
            f"{v_filter};"
            + ";".join(gain_filters)
            + f";{mix_labels}amix=inputs={len(audio_stems)}:normalize=0,"
            f"alimiter=limit=0.97[aout]"
        )
        map_args = ["-map", "[vout]", "-map", "[aout]"]
    elif audio_layout == "dual-track":
        gain_filters = [
            f"[{idx}:a]volume={gain}[a_{name}]"
            for idx, (name, _, gain) in enumerate(audio_stems, start=1)
        ]
        mix_labels = "".join(f"[a_{name}]" for name, _, _ in audio_stems)
        filter_complex = (
            f"{v_filter};"
            f"[1:a]volume={inst_vol},alimiter=limit=0.97[a_off];"
            + ";".join(gain_filters)
            + f";{mix_labels}amix=inputs={len(audio_stems)}:normalize=0,"
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
        left_stems = [
            (idx, name, gain)
            for idx, (name, _, gain) in enumerate(audio_stems, start=1)
            if name != "voc"
        ]
        voc_idx = len(audio_stems)
        left_gains = [
            f"[{idx}:a]volume={gain}[a_{name}]" for idx, name, gain in left_stems
        ]
        if len(left_stems) > 1:
            left_labels = "".join(f"[a_{name}]" for _, name, _ in left_stems)
            left_chain = (
                ";".join(left_gains)
                + f";{left_labels}amix=inputs={len(left_stems)}:normalize=0,"
                "pan=mono|c0=0.5*c0+0.5*c1[a_left]"
            )
        else:
            left_chain = f"[1:a]volume={inst_vol},pan=mono|c0=0.5*c0+0.5*c1[a_left]"
        filter_complex = (
            f"{v_filter};{left_chain};"
            f"[{voc_idx}:a]volume={vocal_vol},pan=mono|c0=0.5*c0+0.5*c1[a_right];"
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
        ["guide.mid", "guide.wav", "vocals_filtered.wav"],
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
        "--filtered-vocal-vol",
        type=parse_volume,
        default=0.0,
        help="Softened/de-sung filtered vocal guide volume level",
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
        ("--filtered-vocal-vol", args.filtered_vocal_vol),
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
        "Extracting guide melody and filtered vocals",
        lambda f: stage5_guide(
            workspace,
            args.guide_timbre,
            manifest,
            f,
            guide_vol=args.guide_vol,
            filtered_vocal_vol=args.filtered_vocal_vol,
        ),
        skip_reason=(
            "guide and filtered vocals muted"
            if args.guide_vol == 0.0
            and args.filtered_vocal_vol == 0.0
            and force_idx != STAGES.index("guide")
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
            filtered_vocal_vol=args.filtered_vocal_vol,
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
