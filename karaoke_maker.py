"""
karaoke_pipeline.py - Japanese Karaoke Generator
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import fugashi
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from basic_pitch.inference import predict
import pykakasi
import stable_whisper


def run_cmd(cmd: list[str]):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def isolate_stems(audio_file: Path, out_dir: Path) -> tuple[Path, Path]:
    print("[1/5] Isolating vocal stem via Demucs...")
    run_cmd(["demucs", "-n", "htdemucs", "-o", str(out_dir), str(audio_file)])
    
    stem_dir = out_dir / "htdemucs" / audio_file.stem
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "instrumental.wav"

    if not instrumental.exists():
        run_cmd([
            "ffmpeg", "-y",
            "-i", str(stem_dir / "drums.wav"),
            "-i", str(stem_dir / "bass.wav"),
            "-i", str(stem_dir / "other.wav"),
            "-filter_complex", "amix=inputs=3:normalize=0",
            str(instrumental)
        ])
    return vocals, instrumental


def build_guide_melody(vocals_path: Path, out_wav: Path, soundfont: Path | None):
    print("[2/5] Synthesizing guide melody line...")
    _, midi_data, _ = predict(
        str(vocals_path),
        onset_threshold=0.6,
        frame_threshold=0.35,
        minimum_note_length=70,
        melodia_trick=True,
    )

    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=500000, time=0))
    # General MIDI 11: Vibraphone
    track.append(Message("program_change", program=11, time=0))

    events = []
    for inst in midi_data.instruments:
        for n in inst.notes:
            if n.end - n.start >= 0.07:
                events.append((n.start, "note_on", n.pitch, 100))
                events.append((n.end, "note_off", n.pitch, 0))

    # Monophonic clamping
    events.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))
    last_time = 0.0
    for t, ev, pitch, vel in events:
        delta = max(0.0, t - last_time)
        ticks = int(mido.second2tick(delta, 480, 500000))
        track.append(Message(ev, note=pitch, velocity=vel, time=ticks))
        last_time = t

    mid_file = out_wav.with_suffix(".mid")
    mid.save(str(mid_file))

    if soundfont and soundfont.exists():
        run_cmd(["fluidsynth", "-ni", str(soundfont), str(mid_file), "-F", str(out_wav), "-r", "44100"])
    else:
        # Fallback silent track if no soundfont is present
        run_cmd(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "5", str(out_wav)])


def get_furigana_morae(text: str) -> list[tuple[str, str | None]]:
    """Splits text into kanji units and their hiragana readings."""
    tagger = fugashi.Tagger()
    kks = pykakasi.kakasi()
    results = []

    for word in tagger(text):
        surface = word.surface
        if re.search(r"[\u4e00-\u9faf]", surface):
            # Extract reading
            kana = word.feature.kana or ""
            conv = kks.convert(kana)
            hira = "".join(item["hira"] for item in conv) if conv else ""
            results.append((surface, hira if hira else None))
        else:
            results.append((surface, None))
    return results


def format_ass_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def create_ass_karaoke(vocals_path: Path, lyrics_path: Path | None, out_ass: Path):
    print("[3/5] Aligning lyrics and generating synchronized ruby ASS tags...")
    model = stable_whisper.load_model("base")

    if lyrics_path and lyrics_path.exists():
        with open(lyrics_path, "r", encoding="utf-8") as f:
            lyrics_text = f.read().strip()
        result = model.align(str(vocals_path), lyrics_text, language="ja")
    else:
        result = model.transcribe(str(vocals_path), language="ja")

    # ASS Header with Japanese karaoke styling (Bottom row + Top preview row)
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KaraokeLine1,Noto Sans CJK JP,58,&H00FFFFFF,&H0024B6FF,&H00101010,&H80000000,1,0,0,0,100,100,2,0,1,4,1,1,120,120,140,1
Style: KaraokeLine2,Noto Sans CJK JP,58,&H00FFFFFF,&H0024B6FF,&H00101010,&H80000000,1,0,0,0,100,100,2,0,1,4,1,3,120,120,60,1
Style: RubyLine1,Noto Sans CJK JP,26,&H00FFFFFF,&H0024B6FF,&H00101010,&H80000000,1,0,0,0,100,100,1,0,1,2,1,1,120,120,205,1
Style: RubyLine2,Noto Sans CJK JP,26,&H00FFFFFF,&H0024B6FF,&H00101010,&H80000000,1,0,0,0,100,100,1,0,1,2,1,3,120,120,125,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    for idx, seg in enumerate(result.segments):
        line_start = seg.start
        line_end = seg.end
        line_num = 1 if (idx % 2 == 0) else 2

        main_builder = []
        ruby_builder = []

        for w in seg.words:
            w_text = w.word.strip()
            if not w_text:
                continue
            dur_cs = max(1, int(round((w.end - w.start) * 100)))
            pairs = get_furigana_morae(w_text)
            step_cs = max(1, dur_cs // len(pairs))

            for kanji, furi in pairs:
                main_builder.append(f"{{\\kf{step_cs}}}{kanji}")
                # Maintain letter-spacing sync on the ruby layer
                if furi:
                    ruby_builder.append(f"{{\\kf{step_cs}}}{furi}")
                else:
                    # Pad space for kana that have no ruby characters above them
                    pad = " " * len(kanji)
                    ruby_builder.append(f"{{\\kf{step_cs}}}{pad}")

        t_s = format_ass_time(line_start)
        t_e = format_ass_time(line_end)
        
        main_text = "".join(main_builder)
        ruby_text = "".join(ruby_builder)

        # Output both layers with precise coordinate anchors
        events.append(f"Dialogue: 1,{t_s},{t_e},RubyLine{line_num},,0,0,0,,{ruby_text}")
        events.append(f"Dialogue: 2,{t_s},{t_e},KaraokeLine{line_num},,0,0,0,,{main_text}")

    with open(out_ass, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    print(f"[4/5] Generated ASS subtitle file: {out_ass}")


def render_karaoke_video(
    bg_path: Path | None,
    instrumental: Path,
    guide_wav: Path,
    vocals_wav: Path,
    ass_path: Path,
    output_mp4: Path,
    guide_vol: float = 0.3,
    vocal_vol: float = 0.0,
    inst_vol: float = 1.0,
    audio_layout: str = "mix",
):
    print("[5/5] Multiplexing video, audio channels, and subtitle overlay...")
    
    if bg_path and bg_path.exists():
        bg_input = ["-i", str(bg_path)]
    else:
        bg_input = ["-f", "lavfi", "-i", "color=c=0x0D1117:s=1920x1080:r=30"]

    ass_arg = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")

    # Inputs: 0 = video, 1 = instrumental, 2 = guide, 3 = vocals
    inputs = [
        *bg_input,
        "-i", str(instrumental),
        "-i", str(guide_wav),
        "-i", str(vocals_wav),
    ]

    if audio_layout == "mix":
        # Hardcode relative volume balance into a single master stereo track
        filter_complex = (
            f"[1:a]volume={inst_vol}[a_inst];"
            f"[2:a]volume={guide_vol}[a_guide];"
            f"[3:a]volume={vocal_vol}[a_voc];"
            f"[a_inst][a_guide][a_voc]amix=inputs=3:duration=first:dropout_transition=2[aout];"
            f"[0:v]ass={ass_arg}[vout]"
        )
        map_args = ["-map", "[vout]", "-map", "[aout]"]

    elif audio_layout == "dual-track":
        # Stream 0: Clean instrumental only
        # Stream 1: Mixed track with custom guide and vocal volumes
        filter_complex = (
            f"[1:a]volume={inst_vol}[a_inst];"
            f"[2:a]volume={guide_vol}[a_guide];"
            f"[3:a]volume={vocal_vol}[a_voc];"
            f"[a_inst][a_guide][a_voc]amix=inputs=3:duration=first[a_mix];"
            f"[0:v]ass={ass_arg}[vout]"
        )
        map_args = [
            "-map", "[vout]",
            "-map", "1:a",      # Audio Track 1: Off-vocal / Pure Instrumental
            "-map", "[a_mix]",  # Audio Track 2: Guide + Vocal assistance
            "-metadata:s:a:0", "title=Instrumental (Off-Vocal)",
            "-metadata:s:a:1", "title=Karaoke Mix (Guide + Vocal)",
        ]

    elif audio_layout == "split-lr":
        # Traditional karaoke hardware panning:
        # Left channel = Instrumental + Guide
        # Right channel = Original Vocals
        filter_complex = (
            f"[1:a]volume={inst_vol}[a_inst];"
            f"[2:a]volume={guide_vol}[a_guide];"
            f"[a_inst][a_guide]amix=inputs=2[a_left_mono];"
            f"[3:a]volume={vocal_vol}[a_right_mono];"
            f"[a_left_mono][a_right_mono]join=inputs=2:channel_layout=stereo[aout];"
            f"[0:v]ass={ass_arg}[vout]"
        )
        map_args = ["-map", "[vout]", "-map", "[aout]"]

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        *map_args,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "320k",
        "-shortest",
        str(output_mp4),
    ]
    run_cmd(cmd)
    print(f"\nRendered video with custom audio balance: {output_mp4}")


def main():
    parser = argparse.ArgumentParser(description="Japanese Karaoke Generator CLI")
    parser.add_argument("song", help="Input song file (.mp3, .flac, .wav)")
    parser.add_argument("--lyrics", "-l", help="Path to plain text lyrics file (for forced alignment)", default=None)
    parser.add_argument("--bg", help="Background video/image", default=None)
    parser.add_argument("--output", "-o", default="karaoke_output.mp4", help="Output MP4 file")
    parser.add_argument(
        "--soundfont", "-sf",
        default="/usr/share/sounds/sf2/default-GM.sf2",
        help="SoundFont (.sf2) path for guide melody synth",
    )
    parser.add_argument("--guide-vol", type=float, default=0.3, 
                    help="Volume multiplier for guide melody (0.0 to mute, 1.0 full)")
    parser.add_argument("--vocal-vol", type=float, default=0.0, 
                    help="Volume multiplier for original backing/lead vocal bleed (0.0 to mute, 1.0 full)")
    parser.add_argument("--inst-vol", type=float, default=1.0, 
                    help="Volume multiplier for instrumental backing track")
    parser.add_argument("--audio-layout", choices=["mix", "dual-track", "split-lr"], default="mix",
                    help="'mix' hardcodes volumes; 'dual-track' keeps stems as separate selectable tracks; 'split-lr' pans vocals to R and inst/guide to L")
    args = parser.parse_args()

    song_path = Path(args.song).resolve()
    if not song_path.exists():
        sys.exit(f"Song file not found: {song_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        vocals, inst = isolate_stems(song_path, tmp)
        
        guide_wav = tmp / "guide.wav"
        build_guide_melody(vocals, guide_wav, Path(args.soundfont))
        
        ass_path = tmp / "karaoke.ass"
        lyrics_in = Path(args.lyrics).resolve() if args.lyrics else None
        create_ass_karaoke(vocals, lyrics_in, ass_path)

        render_karaoke_video(
            bg_path=Path(args.bg).resolve() if args.bg else None,
            instrumental=inst,
            guide_wav=guide_wav,
            ass_path=ass_path,
            output_mp4=Path(args.output).resolve(),
        )


if __name__ == "__main__":
    main()
