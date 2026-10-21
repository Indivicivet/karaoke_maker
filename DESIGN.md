# Japanese karaoke video generator specification

A Python command-line pipeline that turns a music file into an arcade-style (DAM or JOYSOUND) karaoke MP4 video with synced lyrics, furigana (ruby text), and an optional guide melody chime.

## Requirements

### Execution and workflow
- Single-command execution: Running `python karaoke_maker.py "song.mp3"` with only an audio file runs end-to-end and produces a completed MP4.
- Song workspace folder: Each run creates a dedicated output folder for the song containing the final video and all intermediate artifacts (stems, extracted MIDI, generated ASS subtitles, and lyrics).
- Resume and manual edit capability: Stages check for existing artifacts in the workspace before computing. If an artifact exists (or was manually edited), the pipeline picks it up and continues without re-running completed steps.
- Lyrics handling: If a lyrics file is not provided, the pipeline transcribes the vocals automatically. If a lyrics file is provided or already exists in the song folder, it is used directly and never overwritten.

### Lyrics and display
- Japanese arcade styling: Dual-line display mirroring DAM and JOYSOUND layouts. The active singing line wipes progressively while the upcoming line sits in an inactive preview state.
- Progressive color fill: A continuous horizontal color wipe over both the text body and outline synchronized with vocal delivery.
- Kanji and furigana: Full support for kanji with ruby text (furigana) centered directly above the corresponding base characters, with synchronized wipe timings.

### Audio stem balance and volume input
- Independent controls: Support separate levels for backing music, guide melody chime, and original vocal bleed.
- Flexible volume input: Volume flags accept ratios (`0.3`), percentages (`30%`), and decibels (explicit `-6dB` or negative numbers like `-10` treated as dB).
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

## Subtitle generation details (ASS)

- Subtitle format: Advanced SubStation Alpha (.ass v4.00+).
- Fill tag: Progressive fill is implemented using `\kf<duration_cs>` overrides across primary and secondary fill styles.
- Ruby positioning: ASS lacks a native ruby tag. Whitespace padding drifts on proportional fonts; production generation requires calculating bounding-box widths per kanji group (e.g. via font metrics) and placing furigana using absolute `\pos(x, y)` coordinates, or compiling through an Aegisub-compatible template pass.

## Audio mixing implementation

### FFmpeg layouts
- Master mix (`--audio-layout mix`): Merges stems into one stereo stream via `amix=inputs=3:normalize=0`. Disabling normalization preserves absolute input volume multipliers.
- Dual-track (`--audio-layout dual-track`): Muxes stream 0 as clean instrumental (`1:a`) and stream 1 as the guide/vocal blend (`[a_mix]`).
- Left/right split (`--audio-layout split-lr`): Pans instrumental and guide chime to channel 0 (left) and vocal guide to channel 1 (right) via `join=inputs=2:channel_layout=stereo`.

### Volume parsing
Input strings convert to linear amplitude multipliers before building FFmpeg filtergraphs:
- Ratios (`1.0`, `0.3`): Used directly.
- Percentages (`30%`): `val / 100`.
- Decibels (`-10dB`, `-10`): `10 ** (dB / 20)`. Zero volume (mute) maps to amplitude `0.0`.

| Option | Flag | Default |
|---|---|---|
| Instrumental level | `--inst-vol` | `1.0` |
| Guide chime level | `--guide-vol` | `0.3` |
| Vocal guide level | `--vocal-vol` | `0.0` |

## Command-line interface

```bash
# Minimal run (auto-transcribes lyrics, creates song workspace, outputs MP4)
python make_karaoke.py "song.mp3"

# Practice video with quiet vocal assist using percentage and decibel values
python make_karaoke.py "song.mp3" -l "lyrics.txt" --guide-vol 40% --vocal-vol -14dB

# Custom video background with dual selectable audio tracks
python make_karaoke.py "song.mp3" --bg "background.mp4" --audio-layout dual-track
```
