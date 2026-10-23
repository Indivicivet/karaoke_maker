# Japanese karaoke video generator specification

A Python command-line pipeline that turns a music file into an arcade-style (DAM or JOYSOUND) karaoke MP4 video with synced lyrics, furigana (ruby text), and an optional guide melody chime.

## Requirements

### Execution and workflow
- Single-command execution: Running `python karaoke_maker.py "song.mp3"` with only an audio file runs end-to-end and produces a completed MP4.
- Must run on windows.
- Clear progress indicator including time taken so far.
- Song workspace folder: Each run creates a dedicated output folder for the song containing the final video and all intermediate artifacts (stems, extracted MIDI, generated ASS subtitles, and lyrics).
- Resume and manual edit capability: Stages check for existing artifacts in the workspace before computing. If an artifact exists (or was manually edited), the pipeline picks it up and continues without re-running completed steps.
- Lyrics handling: If a lyrics file is not provided, the pipeline transcribes the vocals automatically. If a lyrics file is provided or already exists in the song folder, it is used directly and never overwritten.
- Online lyrics aggregation (TODO): Query online lyric databases (e.g. Genius, LRCLIB, Spotify, or Japanese lyric sites like Uta-Net) using song metadata to fetch reference lyrics before alignment when no local lyrics file is supplied.

### Lyrics and display
- Japanese arcade styling: Dual-line display mirroring DAM and JOYSOUND layouts. The active singing line wipes progressively while the upcoming line sits in an inactive preview state.
- Progressive color fill: A continuous horizontal color wipe over both the text body and outline synchronized with vocal delivery.
- Kanji and furigana: Full support for kanji with ruby text (furigana) centered directly above the corresponding base characters, with synchronized wipe timings.
- Visual countdowns and break indicators: Visual lead-in cues (3-2-1 countdown dots) before lines begin after breaks and at the start of songs so singers can anticipate vocal entries. For long instrumental breaks (>10 seconds), both slots are cleared and an instrumental break timer renders showing the remaining duration until the next vocal entry.
- Duet and multi-singer support (Future exploratory TODO): Support distinct text styling and colors for multiple singers. This will require manual singer annotation syntax in `lyrics.txt` (e.g. `[Singer1]`, `[Singer2]`) or vocal diarization.

### Audio stem balance and volume input
- Independent controls: Support separate levels for backing music, guide melody chime, and original vocal bleed.
- Flexible volume input: Volume flags accept ratios (`0.3`), percentages (`30%`), and decibels (explicit `-6dB` or negative numbers like `-10` treated as dB).
- Audio layouts: Support a combined stereo master mix, a dual-track container with selectable audio streams (pure off-vocal vs. guide assist), and a left/right hardware split (instrumental and guide on left, vocals on right).

### Video backgrounds
- Default background: Automatically generate and cache a reusable 10-second looping motion background (such as subtle floating particles common in Japanese YouTube karaoke videos) to use whenever no background is supplied.
- Custom backgrounds: Support user-supplied background video files and still images.
- Future extension (TODO): Generate mood- and tempo-adaptive particle backgrounds that dynamically react to song sections.

## System requirements and dependencies

Everything the pipeline needs beyond the prerequisites below (separation models, Whisper weights, the subtitle font, the default background) is downloaded or generated automatically on first use and cached in `<repo>/cache/`. No SoundFont or other manually sourced asset is required.

### Repo-local writes only
The pipeline writes only inside the repo folder (`<repo>` = the directory containing `karaoke_maker.py`). Input songs, lyrics files and backgrounds are read and never modified. Several libraries cache to the user profile by default, so every cache location is redirected:

| Cache | Redirect |
|---|---|
| Default background, subtitle font | `cache/bg/`, `cache/fonts/` (our own paths) |
| `audio-separator` models | `Separator(model_file_dir="cache/models/separator")` |
| Whisper weights | `stable_whisper.load_model(..., download_root="cache/models/whisper")` |
| Any other torch / Hugging Face downloads | Set `TORCH_HOME=cache/torch` and `HF_HOME=cache/hf` in `os.environ` at startup, before importing torch or the libraries that use it |

Gotcha: the environment variables have to be set before the first `import torch`, because some libraries read them when they are imported. `cache/` and `songs/` are listed in `.gitignore`.

### Prerequisites
- Python 3.11+ (64-bit). On 3.14 there is no prebuilt Windows wheel for `diffq-fixed` (an `audio-separator` dependency), so it has to be compiled with Visual Studio Build Tools. `audio-separator` also excludes 3.14.1 specifically.
- PyTorch `torch==2.11.0` + `torchaudio==2.11.0`, both from the official PyTorch index (CUDA build if an NVIDIA GPU is available; the CPU build works but separation and Whisper are much slower). The pin exists because torchaudio's newest release is 2.11 while torch is newer, and `stable-ts` and `torchcrepe` both require torchaudio, so the two must be installed as a matching pair.
- FFmpeg on `PATH`, built with libass (e.g. the gyan.dev Windows builds via `winget install Gyan.FFmpeg`). At startup the pipeline runs `ffmpeg -hide_banner -filters` and aborts with an install hint if the `ass` filter is missing.

### Python packages
- `audio-separator`: Vocal/instrumental separation using UVR-family models (Mel-Band RoFormer / BS-RoFormer by default; Demucs and MDX-Net models are also selectable). Models are downloaded on first use.
- `stable-ts`: Whisper transcription (no lyrics supplied) and Whisper-based forced alignment of supplied lyrics to the vocal stem.
- `torchcrepe`: Monophonic f0 (pitch) tracking of the vocal stem with a periodicity (voicing confidence) output.
- `mido`: Writes and reads the editable guide melody MIDI file.
- `fugashi` with `unidic-lite`: Tokenizes Japanese text and supplies kana readings for kanji.
- `fonttools`: Reads glyph advance widths and OS/2 metrics from the subtitle font for layout.
- `numpy`, `soundfile`: Audio arrays, guide chime synthesis, WAV I/O, background frame generation.

> [!NOTE]
> Technology changes from the earlier draft, and why:
> - Demucs → `audio-separator`. The `demucs` package has not had a release since 2023 and saves audio through `torchaudio`; from torchaudio 2.9 the I/O backend moved to TorchCodec, which commonly fails to load on Windows. `audio-separator` is maintained, ships Windows wheels, gives access to RoFormer models with better vocal isolation, and still offers Demucs models if wanted.
> - Basic Pitch → `torchcrepe`. Basic Pitch is a polyphonic note transcriber (not an f0 tracker), it declares support only for Python ≤ 3.11, and it pulls in a TensorFlow/ONNX runtime alongside PyTorch. `torchcrepe` is a PyTorch f0 tracker with Viterbi decoding, so it reuses the torch install already needed for separation and Whisper.
> - FluidSynth + SoundFont → built-in numpy chime synthesis. FluidSynth is a separate Windows install and the SoundFont had no specified source, so a bare `python karaoke_maker.py song.mp3` could not run end-to-end. A short additive-synthesis mallet tone (vibraphone- or glockenspiel-like) is enough for a guide melody and needs no external assets. `guide.mid` is still written, so users can render it in a DAW if they want a different sound.
> - `pykakasi` removed. Converting katakana to hiragana is a fixed Unicode offset (U+30A1–U+30F6 → U+3041–U+3096); a dependency is not needed for it.
> - `fonttools` added. Ruby placement and the outline-inclusive wipe both need glyph advance widths, which the earlier draft required but provided no tool for.

## Song workspace

Default location: `<repo>/songs/<song_stem>_karaoke/` (override with `--workspace`, which must resolve to a path inside the repo).

| Artifact | Producer | Manually editable | Notes |
|---|---|---|---|
| `source.wav` | Stage 1 | no | Input decoded by FFmpeg to 44.1 kHz stereo float WAV. All later stages read only WAV, which avoids codec and non-ASCII-path problems in individual libraries. |
| `vocals.wav`, `instrumental.wav` | Stage 2 | yes (replace) | |
| `lyrics.txt` | user / Stage 3 | yes | One display line per text line. Blank lines are ignored. Optional ruby overrides `{base|reading}`. |
| `alignment.json` | Stage 3 | yes | Start/end times (seconds) of aligned chunks, each mapped to a character range of a lyric line. |
| `lyrics_ruby.txt` | Stage 4 | yes | `lyrics.txt` with every kanji group annotated as `{本気|ほんき}`; edit readings here (e.g. `{本気|まじ}`). |
| `karaoke.ass` | Stage 4 | yes | |
| `guide.mid` | Stage 5 | yes | Monophonic guide melody. |
| `guide.wav` | Stage 5 | yes (replace) | |
| `<song_stem>_karaoke.mp4` | Stage 7 | n/a | Final output. |
| `manifest.json` | all stages | no | Per-artifact fingerprints (see below). |

### Resume, edit detection and staleness
`manifest.json` records, for every artifact the pipeline writes: a SHA-256 of its upstream inputs plus the stage parameters that affect it (`input_fp`), and a SHA-256 of the file as written (`output_fp`). Before running a stage:

1. Artifact missing → run the stage.
2. Artifact hash ≠ `output_fp` (the user edited or replaced it) → keep it and do not overwrite it. If its `input_fp` is also out of date, print a warning that the edited file may be stale and that deleting it will regenerate it.
3. Artifact unchanged and `input_fp` still matches → skip the stage.
4. Artifact unchanged but `input_fp` no longer matches (an upstream file was edited or a relevant flag changed) → regenerate it.

This makes edits propagate downstream. For example, editing `lyrics.txt` triggers re-alignment, a new `lyrics_ruby.txt`, a new ASS file and a re-render, while separation is left alone. Mix-only flags (`--inst-vol` etc.) affect only Stage 7.

### Lyrics source precedence
1. `-l/--lyrics PATH` given: the file is read and never modified. If the workspace already holds a different `lyrics.txt`, that file is renamed to `lyrics.<YYYYmmdd-HHMMSS>.bak.txt` (not overwritten), and the supplied file is copied in as `lyrics.txt`.
2. Otherwise, an existing workspace `lyrics.txt` is used as-is.
3. Otherwise, Stage 3 transcribes the vocals and writes `lyrics.txt` (one line per Whisper segment) together with `alignment.json`, so that later edits to the text can trigger re-alignment.

## Architecture and execution stages

```
Input song (any FFmpeg-readable audio) [+ optional lyrics.txt]
  │
  ├── [Stage 1] Decode (FFmpeg) → source.wav
  │
  ├── [Stage 2] Stem separation (audio-separator, RoFormer vocal model)
  │     ├── vocals.wav
  │     └── instrumental.wav
  │
  ├── [Stage 3] Lyrics and alignment (stable-ts / Whisper, language="ja")
  │     ├── No lyrics: transcribe vocals.wav → lyrics.txt + alignment.json
  │     └── Lyrics:    model.align(vocals.wav, text, original_split=True) → alignment.json
  │
  ├── [Stage 4] Furigana and subtitle layout (fugashi + fontTools)
  │     ├── Readings for kanji groups (okurigana stripped) → lyrics_ruby.txt
  │     ├── Mora segmentation and per-mora timing from alignment.json
  │     ├── Glyph-advance layout of base text and ruby
  │     └── karaoke.ass (dual-line layout, clip-based wipe)
  │
  ├── [Stage 5] Guide melody (torchcrepe + numpy synth)
  │     ├── f0 + periodicity from vocals.wav (10 ms hop)
  │     ├── Note segmentation (+ re-trigger at chunk onsets), drop notes < 70 ms
  │     ├── guide.mid
  │     └── guide.wav (synthesized mallet chime)
  │
  ├── [Stage 6] Background (cached default loop, or user video/image)
  │
  └── [Stage 7] Mix and render (FFmpeg + libass)
        ├── Stem volume balancing per audio layout
        ├── Subtitle burn-in via `ass` filter with `fontsdir`
        └── Output: 1920×1080 H.264 + AAC MP4
```

### Progress reporting
Each stage prints a header such as `[3/7] Aligning lyrics  (total 02:14)` and a footer with that stage's duration, or `skipped (cached)`. Long-running steps show inner progress: the separator and Whisper progress bars are passed through, the background renderer reports frames, and FFmpeg is run with `-progress pipe:1`, whose `out_time_us` is converted into a percentage of the song length. A final summary lists the time for each stage and the total.

### Stage 2: separation
- Default model: a Mel-Band RoFormer vocal model from `audio-separator`'s model list (set with `--sep-model`). The instrumental stem is the model's complementary output.
- Use CUDA when `torch.cuda.is_available()`, otherwise CPU.
- Gotcha: the output file names are chosen by `audio-separator`. Rename them to `vocals.wav` / `instrumental.wav` based on the stem label in the returned file name, not on list order.

### Stage 3: alignment
- Whisper model `--whisper-model`, default `large-v3`. A warning is printed on CPU because it will be slow.
- Input is `vocals.wav`, not the full mix.
- Whisper splits Japanese into tokens and has no spaces to work with, so the "words" that stable-ts returns are token chunks of one or more characters, and a byte-level BPE token can split a kanji. Stage 3 stores these chunks unchanged, after mapping each one onto a character range of the lyric line by walking both strings in order. Dividing a chunk's time among its characters needs the readings, so that is done in Stage 4. This keeps Stage 3 independent of `lyrics_ruby.txt` edits.
- `alignment.json` schema: `{"lines": [{"text": str, "chunks": [{"first": int, "last": int, "start": float, "end": float}]}]}`. `first` and `last` are inclusive character indices into `text`.
- Online lyrics aggregation (TODO): Support querying lyric search providers by track metadata when `--lyrics` is omitted, populating `lyrics.txt` with external reference lyrics to avoid hallucination before running forced alignment.

### Stage 4: furigana and ASS layout

Readings:
- Parse each line with fugashi/UniDic and take `feature.kana` (the orthographic reading). Do not use `pron`, which writes long vowels as `ー`.
- Remove okurigana by matching the kana at the start and end of the surface form against the reading (e.g. `食べる`/`タベル` → `{食|た}べる`), then convert the reading to hiragana. For compounds that cannot be split per character, annotate the whole kanji run as a single group (`{今日|きょう}`).
- `{base|reading}` overrides already present in `lyrics.txt` take precedence. Song lyrics often use non-dictionary readings, so the generated `lyrics_ruby.txt` is the place where users correct them.
- Mora counting: small ゃゅょぁぃぅぇぉゎ merge with the preceding kana; っ, ん and ー each count as one mora.

Timing:
- Character times: a chunk from `alignment.json` that covers several characters has its span divided among them in proportion to their mora counts (taken from the readings above, where a kanji group's morae are shared evenly among its characters). Characters that are not sung (punctuation, spaces) get zero duration.
- A ruby group spans from the start of its first base character to the end of its last. That span is divided equally among the group's ruby morae, unless the base characters carry separate alignment times and the reading can be split per character, in which case each character's own span is used.

Layout (all coordinates use `PlayResX=1920`, `PlayResY=1080`, `ScaledBorderAndShadow: yes`, `WrapStyle: 2`, and `\an7\pos(x,y)` on every event so positions are absolute):
- Font: Noto Sans CJK JP Bold (static OTF), downloaded once from the `notofonts/noto-cjk` repository into `cache/fonts/`, or `--font PATH`. It is passed to libass with `fontsdir`, so rendering does not depend on which fonts are installed on Windows. A static font is used deliberately: variable fonts can render at their default weight in libass.
- Widths come from fontTools `hmtx` advances. Gotcha: libass, like VSFilter, scales the font so that `usWinAscent + usWinDescent` equals the ASS font size, not the em size. Pixel width is therefore `advance * fs / (winAscent + winDescent)`. Kerning is turned off (`Kerning: no` in `[Script Info]`) to match the per-glyph sum.
- Ruby font size is about 0.5× the base size. Each ruby group is centered over the x-extent of its base characters and placed above the base line with its own `\pos` event.
- A line wider than the safe area (1920 − 2×margin) is scaled down horizontally with `\fscx`, and the layout widths are scaled by the same factor.

Dual-line arcade layout:
- Two slots: upper line and lower line, both horizontally centered. Lines alternate between the slots within a continuous singing passage.
- Stanza resets: When the pause between lines is at least 2.5 s, both slots clear after the previous line finishes (leaving a brief blank gap) and the next pair resets to the upper slot (`slot 0`) so static pairs always read top-to-bottom.
- Slot reuse within a stanza: When a slot is reused during continuous singing, the finished line clears ~0.3 s after its last character and the new line waits at least ~0.35 s before appearing (up to 2.5 s before it is sung), avoiding zero-frame in-place text swaps.
- Instrumental break indicator and countdowns: When an instrumental gap exceeds 10 seconds, clear both slots and display an instrumental break indicator showing the seconds remaining until the next phrase. Before the song's first line (when ≥ 2 s) and after pauses ≥ 5 seconds, display 3-2-1 countdown dots above the start of the upcoming line.

Wipe (body and outline):
- `\kf` cannot be used. It sweeps only the fill (Secondary → Primary); the outline keeps `OutlineColour` the whole time, which does not meet the "body and outline" requirement.
- Each line or ruby group is instead drawn twice:
  - Base layer (layer 0): the unsung style (fill, outline and shadow colors), shown for the line's full display time.
  - Sung layer (layer 1): the sung style with different fill and outline colors. It is emitted as one event per timed segment (per character for base text, per mora for ruby). Each event covers only the x-range `[x_k, x_{k+1}]` of its segment and grows its clip with `\clip(x_k,y0,x_k,y1)\t(t_k,t_{k+1},\clip(x_k,y0,x_{k+1},y1))`. The clip rectangles of different segments do not overlap, so there is no double-drawn alpha, and every event holds a single `\t`, which avoids relying on how chained `\t(\clip)` tags behave.
  - Gotcha: `y0`/`y1` and the first and last `x` must be widened by `\bord` + `\shad` (scaled) so that outline pixels outside the advance box are not clipped.
- `--style dam|joysound` chooses the color/outline presets. The preset colors approximate the look of each brand rather than reproducing exact arcade palettes, and can be overridden in the generated ASS styles.

### Stage 5: guide melody
- Load `vocals.wav` with `soundfile`, downmix to mono and resample to 16 kHz before passing the tensor to `torchcrepe.predict`. Do not use `torchcrepe.load.audio`, so that torchaudio I/O is never involved.
- Call `torchcrepe.predict(..., fmin=65, fmax=1100, model="full" on CUDA else "tiny", decoder=viterbi, return_periodicity=True)` with a 10 ms hop (check the keyword name against the installed version).
- Treat a frame as voiced when periodicity > 0.5, the frame RMS is above a gate set relative to the stem's peak RMS, and the frame falls within an aligned lyric chunk window (±50 ms). Bridge brief unvoiced dropouts (≤ 50 ms) when neighboring pitches match, then apply a median filter within each contiguous voiced segment so unvoiced zeros do not pull boundary frames down.
- Segment notes: start a new note when the pitch shifts by ≥ 0.75 semitones from the current note's median and sustains the new pitch for ≥ 120 ms (hysteresis against vibrato and scoops), and also at chunk onsets from `alignment.json` once the current note has lasted ≥ 120 ms. Absorb brief pre-note scoops (< 120 ms) and rapid 1-semitone ornamentation blips into the surrounding note, and drop notes shorter than 70 ms. The output is monophonic by construction.
- Write `guide.mid` with mido (0-based General MIDI program 11 Vibraphone or 9 Glockenspiel, matching `--guide-timbre`, for DAW users).
- Synthesize `guide.wav` according to `--guide-timbre` (default `continuous`):
  - `continuous`: directly drives a warm phase-continuous oscillator (`1× + 0.22×2 + 0.06×3` harmonics) from the smoothed continuous f0 contour and the vocal stem's own smoothed RMS/periodicity envelope, avoiding discrete note attacks or semitone quantization.
  - `vibraphone` / `glockenspiel`: synthesizes discrete notes from `guide.mid` (so that MIDI edits are honored) as a warm additive mallet tone, with softened raised-cosine attack and attenuated overtones on legato transitions. All guide outputs are peak-normalized to −1 dBFS at the length of `source.wav`.

### Stage 6: background
- No `--bg`: use `cache/bg/bg_default_1080p30.mp4`. If it is not there, generate it by rendering 300 frames (10 s at 30 fps) of soft Gaussian particles over a dark vertical gradient in numpy and piping raw RGB to FFmpeg (`libx264`, `yuv420p`).
  - Seamless loop: each particle's x and y are periodic in t with a period of exactly 10 s (integer numbers of wrap-around traversals and sinusoidal drift with integer cycle counts), and the brightness twinkle is also 10 s-periodic, so frame 300 is identical to frame 0.
- Video `--bg`: read with `-stream_loop -1`.
- Image `--bg` (chosen by extension: `.png .jpg .jpeg .webp .bmp`): read with `-loop 1 -framerate 30`.
- All backgrounds pass through `scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30`, and the output length is fixed with `-t <song duration>` taken from `source.wav`. `-shortest` is not used because it is unreliable with looped inputs and filtergraphs.

## Subtitle generation details (ASS)

- Subtitle format: Advanced SubStation Alpha (.ass v4.00+).
- Fill: clip-based two-layer wipe as described in Stage 4. `\kf` is not used because it does not wipe the outline.
- Ruby positioning: absolute `\pos(x, y)` per ruby group from fontTools glyph metrics, using the libass font-size convention above. Whitespace padding is not used because it drifts with proportional fonts.

## Audio mixing implementation

Inputs to Stage 7: `[1]` instrumental, `[2]` guide, `[3]` vocals (input `[0]` is the background). Each stem is first scaled by its parsed linear gain with `volume=<g>`.

### FFmpeg layouts
- Master mix (`--audio-layout mix`, default): `amix=inputs=3:normalize=0`. Disabling normalization keeps the absolute gain multipliers. This is followed by `alimiter=limit=0.97`, because the unnormalized sum can go over full scale.
- Dual-track (`--audio-layout dual-track`):
  - Stream 0, "Off vocal": the instrumental alone at `--inst-vol`.
  - Stream 1, "Guide": instrumental + guide + vocals, mixed as above.
  - Both streams are tagged with `-metadata:s:a:N title=...` and stream 0 is marked `-disposition:a:0 default`. Many players, and web uploads, only play the default stream.
- Left/right split (`--audio-layout split-lr`):
  - Left = mono(instrumental + guide); right = mono(vocals). Each side is downmixed with `pan=mono|c0=0.5*c0+0.5*c1` and the two are combined with `join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR`, then limited.
  - In this layout `--vocal-vol` defaults to `1.0` instead of `0.0`, because the right channel is meant to carry the vocal for the hardware balance knob. An explicit flag still wins.

### Volume parsing
Input strings convert to linear amplitude multipliers before building FFmpeg filtergraphs:
- Ends in `dB` (case-insensitive, sign optional, e.g. `-6dB`, `+3dB`, `0dB`): `10 ** (dB / 20)`.
- Ends in `%`: `val / 100`.
- Bare negative number (`-10`): treated as dB.
- Bare non-negative number (`1.0`, `0.3`, `0`): ratio, used directly. `0` therefore mutes, while `0dB` is unity gain.
- `mute` or `-infdB`: `0.0`.
- Anything else, or a negative ratio/percentage, is an argparse error. Gains above 1.0 are allowed (the limiter stops clipping) and print a warning.

| Option | Flag | Default |
|---|---|---|
| Instrumental level | `--inst-vol` | `1.0` |
| Guide chime level | `--guide-vol` | `0.3` |
| Vocal guide level | `--vocal-vol` | `0.0` (`1.0` with `split-lr`) |

### Encoding
`libx264 -crf 18 -preset medium -pix_fmt yuv420p -r 30`, `aac -b:a 192k`, `-movflags +faststart`.

### Windows gotchas
- The path inside the `ass=` filter argument must be escaped at the filter level. Convert `\` to `/`, escape `:` as `\:` and `'` as `\'`, then wrap the result in single quotes (`ass='C\:/songs/曲/karaoke.ass':fontsdir='C\:/Users/...'`). Pass arguments as a list to `subprocess.run` (no shell) so that non-ASCII paths are passed through unchanged.
- Run separation and Whisper inside `if __name__ == "__main__":` because Windows starts worker processes by spawning them.

## Command-line interface

```bash
# Minimal run (auto-transcribes lyrics, creates song workspace, outputs MP4)
python karaoke_maker.py "song.mp3"

# Practice video with quiet vocal assist using percentage and decibel values
python karaoke_maker.py "song.mp3" -l "lyrics.txt" --guide-vol 40% --vocal-vol -14dB

# Custom video background with dual selectable audio tracks
python karaoke_maker.py "song.mp3" --bg "background.mp4" --audio-layout dual-track
```

| Flag | Default | Purpose |
|---|---|---|
| `input` (positional) | — | Song audio file. |
| `-l/--lyrics` | none | Lyrics text file (see precedence rules). |
| `--workspace` | `songs/<stem>_karaoke` | Song workspace folder (inside the repo). |
| `--bg` | cached default loop | Background video or image. |
| `--audio-layout` | `mix` | `mix`, `dual-track` or `split-lr`. |
| `--inst-vol`, `--guide-vol`, `--vocal-vol` | see table | Stem levels. |
| `--style` | `dam` | `dam` or `joysound` color/layout preset. |
| `--guide-timbre` | `continuous` | `continuous`, `vibraphone`, or `glockenspiel`. |
| `--whisper-model` | `large-v3` (or `medium` if <6GB VRAM) | Whisper model name for stable-ts. |
| `--language` | `auto` | Spoken language code for Whisper (`auto`, `ja`, `en`, etc.). Auto-detects from vocals or lyrics. |
| `--sep-model` | RoFormer vocal model | `audio-separator` model file name. |
| `--font` | `cache/fonts/` Noto Sans CJK JP Bold | Subtitle font file. |
| `--force STAGE` | none | Regenerate this stage and everything after it, even edited artifacts (prompts for confirmation before discarding edits). |

## Verification checklist
- Volume parser unit tests covering every form in the table, including `0` vs `0dB`.
- Ruby splitter unit tests: `食べる`, `今日`, `お願い`, `繰り返す`, kana-only lines, `{base|reading}` overrides.
- Metric calibration: render a single test line with the `ass` filter, measure the drawn bounding box from the frame, and check it against the fontTools prediction (within ±2 px).
- Loop check: decode frames 0 and 299 of the default background and confirm the step between them is no larger than between neighboring frames.
- End-to-end smoke test on a short (~30 s) clip: `python karaoke_maker.py clip.mp3` with no other flags produces an MP4 with no manual steps. Re-running skips every stage. Editing `lyrics_ruby.txt` re-runs only Stage 4 and Stage 7.
