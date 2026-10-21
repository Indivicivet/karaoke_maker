"""Japanese karaoke video generator pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
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


def map_words_to_line_chunks(line_text: str, words: list) -> list[dict]:
    """Walks line_text and aligned words in order, returning chunk dicts."""
    chunks = []
    cursor = 0
    n = len(line_text)

    for w in words:
        w_str = w.word.strip()
        if not w_str:
            continue
        start_t = float(w.start)
        end_t = float(w.end)

        while cursor < n and line_text[cursor].isspace():
            cursor += 1
        if cursor >= n:
            break

        idx = line_text.find(w_str, cursor)
        if idx != -1 and idx - cursor <= 8:
            first = idx
            last = idx + len(w_str) - 1
            cursor = last + 1
        else:
            # Byte-level BPE or minor mismatch: match characters greedily
            first = cursor
            matched = 0
            for ch in w_str:
                if cursor < n and line_text[cursor] == ch:
                    cursor += 1
                    matched += 1
                elif cursor + 1 < n and line_text[cursor + 1] == ch:
                    cursor += 2
                    matched += 1
            if matched == 0:
                cursor = min(n, cursor + max(1, len(w_str)))
            last = max(first, cursor - 1)

        if chunks and first <= chunks[-1]["last"]:
            first = chunks[-1]["last"] + 1
        if first < n:
            last = min(n - 1, max(first, last))
            chunks.append(
                {
                    "first": first,
                    "last": last,
                    "start": round(start_t, 4),
                    "end": round(max(start_t, end_t), 4),
                }
            )

    return chunks


def has_japanese_text(text: str) -> bool:
    """Checks if text contains Hiragana, Katakana, or CJK Kanji."""
    return any(
        (0x3040 <= ord(c) <= 0x309F)
        or (0x30A0 <= ord(c) <= 0x30FF)
        or (0x4E00 <= ord(c) <= 0x9FFF)
        for c in text
    )


def detect_audio_language(model, vocals_wav: Path) -> str:
    """Detects spoken language from vocal stem audio using Whisper."""
    import whisper

    audio = whisper.load_audio(str(vocals_wav))
    mel = whisper.log_mel_spectrogram(audio, model.dims.n_mels)
    mel_seg = (
        whisper.pad_or_trim(mel, 3000)
        .to(model.device)
        .to(next(model.parameters()).dtype)
    )
    _, probs = model.detect_language(mel_seg)
    detected = max(probs, key=probs.get)
    print(f"  Detected audio language: {detected} (confidence: {probs[detected]:.1%})")
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

    if has_user_lyrics or (lyrics_txt.exists() and not force):
        upstream = [vocals_wav, lyrics_txt]
        artifacts = [alignment_json]
    else:
        upstream = [vocals_wav]
        artifacts = [lyrics_txt, alignment_json]

    input_fp = compute_fingerprint(
        upstream, {"stage": 3, "whisper_model": whisper_model, "language": language}
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

    if lyrics_txt.exists() and (has_user_lyrics or not force):
        raw_lines = [
            ln.strip()
            for ln in lyrics_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        clean_lines = [strip_ruby_overrides(ln) for ln in raw_lines]
        joined_text = "\n".join(clean_lines)

        if resolved_lang is None:
            if has_japanese_text(joined_text):
                resolved_lang = "ja"
            else:
                resolved_lang = detect_audio_language(model, vocals_wav)

        result = model.align(
            str(vocals_wav),
            joined_text,
            language=resolved_lang,
            original_split=True,
        )
        out_lines = []
        for idx, clean_ln in enumerate(clean_lines):
            if idx < len(result.segments):
                seg = result.segments[idx]
                chunks = map_words_to_line_chunks(clean_ln, seg.words)
            else:
                chunks = []
            out_lines.append({"text": clean_ln, "chunks": chunks})
    else:
        if resolved_lang is None:
            resolved_lang = detect_audio_language(model, vocals_wav)

        transcribe_kwargs = {"language": resolved_lang}
        if device == "cuda" and total_vram_gb < 6.0:
            transcribe_kwargs["fp16"] = False
        result = model.transcribe(str(vocals_wav), **transcribe_kwargs)
        transcribed_lines = []
        out_lines = []
        for seg in result.segments:
            text = seg.text.strip()
            if not text:
                continue
            transcribed_lines.append(text)
            chunks = map_words_to_line_chunks(text, seg.words)
            out_lines.append({"text": text, "chunks": chunks})

        lyrics_txt.write_text("\n".join(transcribed_lines) + "\n", encoding="utf-8")

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

    times: list[tuple[float, float] | None] = [None] * n
    for ch_info in chunks:
        c_first = max(0, min(n - 1, int(ch_info["first"])))
        c_last = max(c_first, min(n - 1, int(ch_info["last"])))
        t0 = float(ch_info["start"])
        t1 = max(t0, float(ch_info["end"]))
        dur = t1 - t0

        weights = [char_mora_weights[i] for i in range(c_first, c_last + 1)]
        total_w = sum(weights)
        cur_t = t0
        for offset, w in enumerate(weights):
            idx = c_first + offset
            if total_w > 0:
                dt = dur * (w / total_w)
            else:
                dt = 0.0
            times[idx] = (cur_t, cur_t + dt)
            cur_t += dt

    # Fill any unaligned characters by interpolating between neighboring chunks
    last_t = chunks[0]["start"] if chunks else 0.0
    for i in range(n):
        if times[i] is not None:
            last_t = times[i][1]
        else:
            next_t = last_t
            for j in range(i + 1, n):
                if times[j] is not None:
                    next_t = times[j][0]
                    break
            times[i] = (
                (last_t, last_t) if char_mora_weights[i] == 0 else (last_t, next_t)
            )

    return [t if t is not None else (0.0, 0.0) for t in times]


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
        [lyrics_ruby_txt, alignment_json, font_path],
        {"stage": "4_ass", "style": style_preset},
    )
    run_ass, kept_ass = check_stage_status([karaoke_ass], ass_input_fp, manifest, force)
    if not run_ass:
        return run_ruby

    metrics = FontMetrics(font_path)
    align_data = json.loads(alignment_json.read_text(encoding="utf-8")).get("lines", [])
    ruby_lines = [
        ln.strip()
        for ln in lyrics_ruby_txt.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

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

    base_fs = 68.0
    ruby_fs = 34.0
    base_bord = 4.0
    ruby_bord = 2.5
    shad = 2.0
    margin_x = 120.0
    safe_width = 1920.0 - 2.0 * margin_x
    slot_y = (780.0, 910.0)
    ruby_gap = 6.0

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2
Kerning: no

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BaseUnsung,{metrics.family_name},{int(base_fs)},{unsung_fill},{unsung_fill},{unsung_out},{shad_col},0,0,0,0,100,100,0,0,1,{base_bord:g},{shad:g},7,0,0,0,1
Style: BaseSung,{metrics.family_name},{int(base_fs)},{sung_fill},{sung_fill},{sung_out},{shad_col},0,0,0,0,100,100,0,0,1,{base_bord:g},{shad:g},7,0,0,0,1
Style: RubyUnsung,{metrics.family_name},{int(ruby_fs)},{unsung_fill},{unsung_fill},{unsung_out},{shad_col},0,0,0,0,100,100,0,0,1,{ruby_bord:g},{shad:g},7,0,0,0,1
Style: RubySung,{metrics.family_name},{int(ruby_fs)},{sung_fill},{sung_fill},{sung_out},{shad_col},0,0,0,0,100,100,0,0,1,{ruby_bord:g},{shad:g},7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    parsed_lines = []
    for idx, r_line in enumerate(ruby_lines):
        plain_text, segments = parse_ruby_line_segments(r_line)
        chunks = align_data[idx].get("chunks", []) if idx < len(align_data) else []
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
    break_threshold = 8.0
    slot_idx = 0
    for i, item in enumerate(parsed_lines):
        if i == 0 or (item["start"] - parsed_lines[i - 1]["end"]) > break_threshold:
            slot_idx = 0
        else:
            slot_idx = 1 - slot_idx
        item["slot"] = slot_idx

    for i, item in enumerate(parsed_lines):
        is_pair_start = i == 0 or (
            item["start"] - parsed_lines[i - 1]["end"] > break_threshold
        )
        if is_pair_start:
            disp_start = max(0.0, item["start"] - 3.0)
            if i > 0:
                disp_start = max(disp_start, parsed_lines[i - 1]["end"] + 0.5)
        elif i == 1 or (
            i >= 2
            and parsed_lines[i - 1]["start"] - parsed_lines[i - 2]["end"]
            > break_threshold
        ):
            disp_start = parsed_lines[i - 1]["disp_start"]
        else:
            prev_same_slot = parsed_lines[i - 2]
            disp_start = min(item["start"], prev_same_slot["end"] + 0.3)

        disp_end = item["end"] + 0.5
        if (
            i + 1 < n_lines
            and (parsed_lines[i + 1]["start"] - item["end"]) <= break_threshold
            and i + 2 < n_lines
            and parsed_lines[i + 2]["slot"] == item["slot"]
        ):
            next_same = parsed_lines[i + 2]
            needed = min(next_same["start"], item["end"] + 0.3)
            disp_end = min(disp_end, needed)

        item["disp_start"] = disp_start
        item["disp_end"] = max(item["end"], disp_end)

    events = []
    for item in parsed_lines:
        plain = item["plain"]
        segments = item["segments"]
        char_times = item["char_times"]
        slot = item["slot"]
        disp_s = item["disp_start"]
        disp_e = item["disp_end"]
        ts_str = format_ass_time(disp_s)
        te_str = format_ass_time(disp_e)

        unscaled_w = metrics.text_width(plain, base_fs, 1.0)
        scale_x = min(1.0, safe_width / unscaled_w) if unscaled_w > 0 else 1.0
        fscx_pct = int(round(scale_x * 100))
        fscx_tag = f"\\fscx{fscx_pct}" if fscx_pct != 100 else ""
        total_w = unscaled_w * scale_x

        line_x = margin_x if slot == 0 else (1920.0 - margin_x - total_w)
        line_y = slot_y[slot]

        char_x = []
        cur_x = line_x
        for ch in plain:
            w = metrics.char_width(ch, base_fs, scale_x)
            char_x.append((cur_x, cur_x + w))
            cur_x += w

        pos_tag = f"{{\\an7\\pos({line_x:.2f},{line_y:.2f}){fscx_tag}}}"
        events.append(
            f"Dialogue: 0,{ts_str},{te_str},BaseUnsung,,0,0,0,,{pos_tag}{plain}"
        )

        pad_base = (base_bord + shad) * scale_x + 2.0
        y0_base = int(math.floor(line_y - base_bord - shad - 4.0))
        y1_base = int(math.ceil(line_y + base_fs + base_bord + shad + 4.0))

        n_chars = len(plain)
        for c_idx in range(n_chars):
            x_left, x_right = char_x[c_idx]
            clip_x0 = (
                int(math.floor(x_left - pad_base)) if c_idx == 0 else int(round(x_left))
            )
            clip_x1 = (
                int(math.ceil(x_right + pad_base))
                if c_idx == n_chars - 1
                else int(round(x_right))
            )
            if clip_x1 <= clip_x0:
                continue

            ct0, ct1 = char_times[c_idx]
            rel_t0 = max(0, int(round((ct0 - disp_s) * 1000)))
            rel_t1 = max(rel_t0 + 1, int(round((ct1 - disp_s) * 1000)))
            wipe_tag = (
                f"{{\\an7\\pos({line_x:.2f},{line_y:.2f}){fscx_tag}"
                f"\\clip({clip_x0},{y0_base},{clip_x0},{y1_base})"
                f"\\t({rel_t0},{rel_t1},\\clip({clip_x0},{y0_base},{clip_x1},{y1_base}))}}"
            )
            events.append(
                f"Dialogue: 1,{ts_str},{te_str},BaseSung,,0,0,0,,{wipe_tag}{plain}"
            )

        # Emit ruby layers
        ruby_y = line_y - ruby_fs - ruby_gap
        y0_ruby = int(math.floor(ruby_y - ruby_bord - shad - 3.0))
        y1_ruby = int(math.ceil(ruby_y + ruby_fs + ruby_bord + shad + 3.0))
        pad_ruby = (ruby_bord + shad) * scale_x + 2.0

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
            r_pos_tag = f"{{\\an7\\pos({r_x:.2f},{ruby_y:.2f}){r_fscx_tag}}}"
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
                rel_mt0 = max(0, int(round((mt0 - disp_s) * 1000)))
                rel_mt1 = max(rel_mt0 + 1, int(round((mt1 - disp_s) * 1000)))

                r_wipe_tag = (
                    f"{{\\an7\\pos({r_x:.2f},{ruby_y:.2f}){r_fscx_tag}"
                    f"\\clip({r_clip0},{y0_ruby},{r_clip0},{y1_ruby})"
                    f"\\t({rel_mt0},{rel_mt1},\\clip({r_clip0},{y0_ruby},{r_clip1},{y1_ruby}))}}"
                )
                events.append(
                    f"Dialogue: 1,{ts_str},{te_str},RubySung,,0,0,0,,{r_wipe_tag}{r_text}"
                )

    karaoke_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    record_artifacts([karaoke_ass], kept_ass, ass_input_fp, manifest, workspace)
    return True


def synthesize_mallet_guide(
    midi_path: Path, out_wav: Path, duration_sec: float, timbre: str, sr: int = 44100
):
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

    total_samples = int(round(duration_sec * sr))
    buf = np.zeros(total_samples, dtype=np.float32)

    if timbre == "glockenspiel":
        partials = [
            (1.0, 1.0, 3.5),
            (2.76, 0.45, 7.0),
            (5.40, 0.25, 11.0),
            (8.93, 0.12, 16.0),
        ]
    else:
        # Vibraphone-likepartials: 1x, 4x, 10x
        partials = [(1.0, 1.0, 2.5), (4.0, 0.35, 6.5), (10.0, 0.10, 14.0)]

    fade_samples = max(1, int(round(0.005 * sr)))
    tail_sec = 0.18

    for t0, t1, midi_note in notes:
        s0 = max(0, int(round(t0 * sr)))
        note_dur = max(0.05, t1 - t0)
        render_dur = min(duration_sec - t0, note_dur + tail_sec)
        if render_dur <= 0 or s0 >= total_samples:
            continue
        n_samp = min(total_samples - s0, int(round(render_dur * sr)))
        if n_samp <= 0:
            continue

        t = np.arange(n_samp, dtype=np.float32) / float(sr)
        f0 = 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))
        tone = np.zeros(n_samp, dtype=np.float32)
        for ratio, amp, decay in partials:
            freq = f0 * ratio
            if freq < sr * 0.45:
                tone += amp * np.sin(2.0 * np.pi * freq * t) * np.exp(-decay * t)

        env = np.ones(n_samp, dtype=np.float32)
        att = min(fade_samples, n_samp)
        env[:att] = np.linspace(0.0, 1.0, att, dtype=np.float32)

        off_idx = min(n_samp, int(round(note_dur * sr)))
        if off_idx < n_samp:
            rel_len = n_samp - off_idx
            rel_curve = np.exp(-np.linspace(0.0, 6.0, rel_len, dtype=np.float32))
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
        {"stage": "5_mid", "timbre": guide_timbre},
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
            fmin=65.0,
            fmax=1100.0,
            model=crepe_model,
            decoder=torchcrepe.decode.viterbi,
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
        voiced = (pd_np > 0.5) & (rms > rms_gate) & (f0_np >= 65.0)

        midi_float = np.zeros(n_frames, dtype=np.float32)
        valid_f0 = np.maximum(f0_np, 1e-3)
        midi_float[voiced] = 69.0 + 12.0 * np.log2(valid_f0[voiced] / 440.0)
        midi_filt = median_filter(midi_float, size=5)
        midi_filt[~voiced] = 0.0

        align_data = json.loads(alignment_json.read_text(encoding="utf-8")).get(
            "lines", []
        )
        onset_frames = set()
        for ln in align_data:
            for ch in ln.get("chunks", []):
                onset_frames.add(int(round(float(ch["start"]) / 0.01)))

        notes = []
        cur_start = None
        cur_frames: list[float] = []

        for i in range(n_frames):
            if not voiced[i] or midi_filt[i] <= 0:
                if cur_start is not None:
                    if (i - cur_start) >= 7:
                        pitch = int(round(float(np.median(cur_frames))))
                        notes.append((cur_start * 0.01, i * 0.01, pitch))
                    cur_start = None
                    cur_frames = []
                continue

            p_round = int(round(float(midi_filt[i])))
            if cur_start is None:
                cur_start = i
                cur_frames = [float(midi_filt[i])]
            else:
                retrigger = i in onset_frames
                cur_med = int(round(float(np.median(cur_frames))))
                if not retrigger and p_round != cur_med:
                    ahead = midi_filt[i : min(n_frames, i + 5)]
                    ahead_v = voiced[i : min(n_frames, i + 5)]
                    if len(ahead) == 5 and np.all(ahead_v):
                        if np.all(np.abs(np.round(ahead) - p_round) == 0):
                            retrigger = True
                if retrigger:
                    if (i - cur_start) >= 7:
                        pitch = int(round(float(np.median(cur_frames))))
                        notes.append((cur_start * 0.01, i * 0.01, pitch))
                    cur_start = i
                    cur_frames = [float(midi_filt[i])]
                else:
                    cur_frames.append(float(midi_filt[i]))

        if cur_start is not None and (n_frames - cur_start) >= 7:
            pitch = int(round(float(np.median(cur_frames))))
            notes.append((cur_start * 0.01, n_frames * 0.01, pitch))

        mid = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        mid.tracks.append(track)
        tempo = 500000
        track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
        program = 9 if guide_timbre == "glockenspiel" else 11
        track.append(Message("program_change", program=program, time=0))

        last_t = 0.0
        for t0, t1, pitch in notes:
            dt_on = max(0.0, t0 - last_t)
            ticks_on = int(round(mido.second2tick(dt_on, 480, tempo)))
            track.append(Message("note_on", note=pitch, velocity=100, time=ticks_on))
            dt_off = max(0.0, t1 - t0)
            ticks_off = int(round(mido.second2tick(dt_off, 480, tempo)))
            track.append(Message("note_off", note=pitch, velocity=0, time=ticks_off))
            last_t = t1

        mid.save(str(guide_mid))
        record_artifacts([guide_mid], kept_mid, mid_fp, manifest, workspace)

    wav_fp = compute_fingerprint(
        [guide_mid, source_wav],
        {"stage": "5_wav", "timbre": guide_timbre},
    )
    run_wav, kept_wav = check_stage_status([guide_wav], wav_fp, manifest, force)
    if not run_wav:
        return run_mid

    info = sf.info(str(source_wav))
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

    input_fp = compute_fingerprint(
        [inst_wav, guide_wav, vocals_wav, karaoke_ass, bg_path, font_path],
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

    cmd = [
        "ffmpeg",
        "-y",
        *bg_in,
        "-i",
        str(inst_wav),
        "-i",
        str(guide_wav),
        "-i",
        str(vocals_wav),
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
        choices=["vibraphone", "glockenspiel"],
        default="vibraphone",
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
    stage_times: list[tuple[str, float, bool]] = []

    def run_stage(idx: int, title: str, fn):
        elapsed_total = time.perf_counter() - t_pipeline_start
        print(f"\n[{idx}/7] {title}  (total {format_duration(elapsed_total)})")
        t0 = time.perf_counter()
        forced = force_idx is not None and (idx - 1) >= force_idx
        ran = fn(forced)
        dt = time.perf_counter() - t0
        if ran:
            print(f"  done in {format_duration(dt)}")
        else:
            print("  skipped (cached)")
        stage_times.append((f"[{idx}/7] {title}", dt, ran))

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
    for label, dt, ran in stage_times:
        status = format_duration(dt) if ran else "skipped (cached)"
        print(f"  {label:<45} {status}")
    print(f"  {'Total':<45} {format_duration(total_time)}")
    print(f"  Output: {out_holder[0]}")


if __name__ == "__main__":
    main()
