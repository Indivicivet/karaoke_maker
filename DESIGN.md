# Japanese karaoke video generator specification

A Python command-line pipeline that turns a music file into an arcade-style (DAM or JOYSOUND) karaoke MP4 video with synced lyrics, furigana (ruby text), and an optional guide melody chime.

## Requirements

### Execution and workflow
- Single-command execution: Running `python karaoke_maker.py "song.mp3"` with only an audio file must run end-to-end and produce a completed MP4.
- Song workspace folder: Each run creates a dedicated output folder for the song containing the final video and all intermediate artifacts (stems, extracted MIDI, generated ASS subtitles, and lyrics).
- Resume and manual edit capability: Stages check for existing artifacts in the workspace before computing. If an artifact exists (or was manually edited), the pipeline picks it up and continues without re-running completed steps.
- Lyrics handling: If a lyrics file is not provided, the pipeline transcribes the vocals automatically. If a lyrics file is provided or already exists in the song folder, it is used directly and never overwritten.

### Lyrics and display
- Japanese arcade styling: Dual-line display mirroring DAM and JOYSOUND layouts. The active singing line wipes progressively while the upcoming line sits in an inactive preview state.
- Progressive color fill: A continuous horizontal color wipe over both the text body and outline synchronized with vocal delivery.
- Kanji and furigana: Full support for kanji with ruby text (furigana) centered directly above the corresponding base characters, with synchronized wipe timings.

### Audio stem balance and volume input
- Independent controls: Support separate levels for backing music, guide melody chime, and original vocal bleed.
- Flexible volume input: Volume flags accept three formats:
  - Ratios: `0`, `0.3`, `1.0`
  - Percentages: `0%`, `30%`, `100%`
  - Decibels: explicit strings like `-6dB` or negative numbers like `-10`, which are parsed as decibel values.
- Audio layouts: Support a combined stereo master mix, a dual-track container with selectable audio streams (pure off-vocal vs. guide assist), and a left/right hardware split (instrumental and guide on left, vocals on right).

### Video backgrounds
- Default background: Automatically generate and cache a reusable 10-second looping motion background (such as subtle floating particles common in Japanese YouTube karaoke videos) to use whenever no background is supplied.
- Custom backgrounds: Support user-supplied background video files and still images.
- Future extension (TODO): Generate mood- and tempo-adaptive particle backgrounds that dynamically react to song sections.

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
  │     ├── Forced alignment against lyric text (kana-mapped morae)
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

### Volume parsing rules
Volume options parse three input types into linear amplitude multipliers:
- Ratios: Floats from 0.0 to 1.0 (or greater) map directly.
- Percentages: `X%` maps to `X / 100`.
- Decibels: Any value ending in `dB` or any negative numeric value (e.g. `-10`) converts via `10 ** (dB / 20)`.

| Option | Accepted syntax | Default | Description |
|---|---|---|---|
| `--inst-vol` | `1.0`, `100%`, `0dB` | `1.0` | Instrumental volume |
| `--guide-vol` | `0.3`, `30%`, `-10dB` | `0.3` | Guide melody chime volume (0 mutes) |
| `--vocal-vol` | `0.0`, `0%`, `-inf` | `0.0` | Vocal bleed volume (0 for clean karaoke) |

## Command-line interface

```bash
# Minimal run (auto-transcribes lyrics, creates song workspace, outputs MP4)
python make_karaoke.py "song.mp3"

# Practice video with quiet vocal assist using percentage and decibel values
python make_karaoke.py "song.mp3" -l "lyrics.txt" --guide-vol 40% --vocal-vol -14dB

# Custom video background with dual selectable audio tracks
python make_karaoke.py "song.mp3" --bg "background.mp4" --audio-layout dual-track
```
