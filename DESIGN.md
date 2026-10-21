Japanese Karaoke Video Generator Pipeline Specification
A modular, single-command Python pipeline to transform an arbitrary input song into a Japanese arcade-style (DAM / JOYSOUND) karaoke MP4.1. System Requirements & External DependenciesCLI Tools & System PackagesFFmpeg: Video compositing, audio mixing/filtering, and ASS subtitle burning (libass enabled).FluidSynth: SoundFont-based MIDI synthesizer for rendering the guide chime audio.General MIDI SoundFont: Standard .sf2 patch set containing standard melodic percussion (e.g., FluidR3_GM.sf2 or default-GM.sf2).Python Dependenciesdemucs: AI source separation (extracts pristine vocal and instrumental stems).stable-ts: Acoustic forced-alignment via Whisper models down to word and mora millisecond boundaries.basic-pitch: Spotify's neural network for monophonic fundamental frequency ($f_0$) pitch tracking to MIDI.mido: MIDI file generation, event sorting, tick quantization, and delta-time parsing.fugashi + unidic-lite: Morphological analysis for Kanji segmentation and accurate reading extraction.pykakasi: Kana-to-Hira conversion and mora splitting.2. Architecture & Execution StagesInput Song (Audio) [+ Optional Plaintext Lyrics]
  │
  ├── [Stage 1] AI Stem Separation (Demucs)
  │     ├── vocals.wav
  │     └── instrumental.wav
  │
  ├── [Stage 2] AI Guide Melody Extraction (Basic Pitch + FluidSynth)
  │     ├── Monophonic pitch tracking ($f_0$)
  │     ├── Noise/breath cutoff (< 70ms rejection) & quantization
  │     ├── General MIDI Patch 11 (Vibraphone / Glockenspiel)
  │     └── Output: guide.wav
  │
  ├── [Stage 3] Syllable Alignment & Furigana Layout (stable-ts + Fugashi)
  │     ├── Acoustic forced alignment to lyrics text
  │     ├── Morphological parse: Kanji compound -> Furigana reading
  │     ├── Sub-syllabic mora timing allocation
  │     └── ASS Generation: Dual-line layout with continuous \kf color fill
  │
  └── [Stage 4] Audio Balancing & Subtitle Burning (FFmpeg)
        ├── Subtitle burning via ass filter
        ├── Independent gain controls (instrumental, guide, vocal stems)
        └── Output: Final 1080p MP4
3. Subtitle Engine Specification (ASS)Styling RulesFormat: Advanced SubStation Alpha (.ass v4.00+).Color Sweep: Continuous horizontal fill wipe across both text fill and outline using the \kf<duration_cs> override tag.Dual-Line Logic:Even lines render to the active position.Odd lines render to the alternate line position in advance as a reading preview, mirroring JOYSOUND / DAM display systems.Furigana / Ruby Centering:Furigana text is rendered above corresponding kanji compounds at roughly 45–50% main font size.Co-timed so that the \kf duration of the kanji compound matches the sum of the swept morae in the furigana row above it.4. Audio Control & Mixing MatrixThe generator supports independent gain controls for all three audio stems prior to final multiplexing:FlagParameterDefaultFunction--inst-volFloat1.0Background music / off-vocal backing level--guide-volFloat0.3Synth bell guide melody level (0.0 mutes entirely)--vocal-volFloat0.0Original vocal bleed level (0.0 for pure karaoke, > 0.0 for vocal assist)Supported Output Audio ConfigurationsMaster Stereo Mix (--audio-layout mix): Single stereo stream blending instrumental, guide melody, and attenuated vocals using FFmpeg's amix filter.Dual-Track Stream (--audio-layout dual-track): Container holds two selectable stereo tracks (Track 1 = pure instrumental off-vocal; Track 2 = karaoke mix with guide and vocal cues).Hardware L/R Split (--audio-layout split-lr): Traditional hardware karaoke panning (Left channel = Instrumental + Guide chime; Right channel = Original Vocals) for physical mixer balancing.5. Command-Line Interface SpecificationBash# Standard pure karaoke video (Instrumental + 30% Guide Chime)
python make_karaoke.py "song.mp3" -l "lyrics.txt" -o "karaoke.mp4"

# Practice mode with low backing vocal assist
python make_karaoke.py "song.mp3" -l "lyrics.txt" --guide-vol 0.4 --vocal-vol 0.2 -o "practice.mp4"

# Custom video background with dual selectable audio tracks
python make_karaoke.py "song.mp3" --bg "background.mp4" --audio-layout dual-track -o "kar