# Japanese karaoke video generator specification

A Python command-line pipeline that turns a music file into an arcade-style (DAM or JOYSOUND) karaoke MP4 video with synced lyrics, furigana (ruby text), and an optional guide melody chime.

## System requirements and dependencies

### System tools
- FFmpeg (compiled with libass support) for audio mixing, video encoding, and burning ASS subtitles.
- FluidSynth for synthesizing MIDI guide melodies through a SoundFont.
- General MIDI SoundFont (`.sf2`), such as FluidR3_GM.sf2 or default-GM.sf2, containing standard melodic percussion (vibraphone or glockenspiel).

### Python packages
- `demucs`: Separates input songs into vocal and backing tracks.
- `stable-ts`: Aligns Whisper speech-to-text models to vocal audio at word and character boundaries.
- `basic-pitch`: Tracks pitch from the isolated vocal stem to generate MIDI notes.
- `mido`: Builds and edits MIDI files, handles tick quantization, and writes program changes.
- `fugashi` with `unidic-lite`: Tokenizes Japanese text and extracts readings for kanji compounds.
- `pykakasi`: Converts readings between katakana and hiragana.

## Architecture and execution stages

```
Input song (audio) [+ optional plain text lyrics]
  │
  ├── [Stage 1] Stem separation (Demucs)
  │     ├── vocals.wav
  │     └── instrumental.wav
  │
  ├── [Stage 2] Guide melody synthesis (Basic Pitch + FluidSynth)
  │     ├── Pitch tracking (f0) from vocals.wav
  │     ├── Note cleanup (rejection under 70ms, monophonic filtering)
  │     ├── General MIDI program 11 (Vibraphone) or 9 (Glockenspiel)
  │     └── Output: guide.wav
  │
  ├── [Stage 3] Syllable alignment and furigana layout (stable-ts + Fugashi)
  │     ├── Forced alignment against lyric text
  │     ├── Morphological parsing to generate furigana for kanji
  │     ├── Mora timing distribution
  │     └── ASS subtitle generation (dual-line layout with continuous \kf sweeps)
  │
  └── [Stage 4] Audio mixing and video rendering (FFmpeg)
        ├── Subtitle burning with libass
        ├── Stem volume balancing (instrumental, guide chime, vocals)
        └── Output: 1080p MP4
```

## Subtitle engine (ASS)

### Styling rules
- Format: Advanced SubStation Alpha (.ass v4.00+).
- Fill animation: Smooth progressive color fill using `\kf<duration_cs>` tags across text and outlines.
- Two-line alternation: Line 1 sits on top (or row A) and line 2 sits on the bottom (or row B). While line 1 is active, line 2 shows upcoming lyrics in advance in inactive grey or white, matching arcade displays. When line 1 finishes, line 3 replaces it before its cue starts.
- Furigana layout: Furigana text sits directly above the associated kanji at roughly 45% to 50% of the main font size. The duration of the kanji wipe matches the total duration of the furigana morae above it. Note: ASS has no native ruby layout tag. Aligning furigana with whitespace padding fails on proportional fonts because kanji and spaces have different character widths. Production implementations need pixel-width measurements per segment or coordinate-anchored `\pos` tags.

## Audio mixing and stem balancing

The pipeline balances three audio streams before muxing:
- Instrumental backing track
- Guide chime melody (synthesized bells)
- Original vocal bleed (optional, useful for practice tracks)

### Mixing options
- Master stereo mix (`--audio-layout mix`): Blends instrumental, guide chime, and vocals into a single stereo track. When using FFmpeg's `amix` filter, `normalize=0` should be set to prevent automatic volume drops when inputs are combined.
- Dual-track container (`--audio-layout dual-track`): Saves two selectable audio streams into the MP4 container. Track 1 is the clean off-vocal backing, and track 2 is the karaoke mix with guide chime and vocal hints.
- Left/right split (`--audio-layout split-lr`): Classic karaoke hardware split. The left channel carries backing music and the guide chime, while the right channel carries the vocal guide, allowing physical mixers to dial in vocals via pan pots.

### Volume options
| Option | Type | Default | Description |
|---|---|---|---|
| `--inst-vol` | float | 1.0 | Instrumental volume multiplier |
| `--guide-vol` | float | 0.3 | Guide melody chime volume (0.0 mutes) |
| `--vocal-vol` | float | 0.0 | Vocal bleed volume (0.0 for clean karaoke, > 0.0 for vocal assist) |

## Command-line interface

```bash
# Standard karaoke video (instrumental + 30% guide chime)
python make_karaoke.py "song.mp3" -l "lyrics.txt" -o "karaoke.mp4"

# Practice video with quiet vocal assist
python make_karaoke.py "song.mp3" -l "lyrics.txt" --guide-vol 0.4 --vocal-vol 0.2 -o "practice.mp4"

# Custom video background with dual selectable audio tracks
python make_karaoke.py "song.mp3" --bg "background.mp4" --audio-layout dual-track -o "karaoke_dual.mp4"
```
