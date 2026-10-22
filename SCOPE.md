# Project Scope and Excluded Features

This document outlines features from existing karaoke tools (such as nomadkaraoke's karaoke-gen) that have been intentionally excluded from karaoke_maker, deferred, or marked out of scope.

## Excluded features

### Guitar chords
Displaying guitar chord charts, chord diagrams, or top-right "now / next" chord boxes is completely out of scope. The project focuses strictly on vocal performance, Japanese arcade layouts (DAM/JOYSOUND), and pitch-guided melody tracking.

### Legacy and alternative export formats
Generating CDG+MP3 (for physical KJ jukeboxes), TXT+MP3, standalone LRC files, or 4K master encodes is out of scope. The standard output is a self-contained 1080p H.264/AAC MP4 video compatible with desktop players, mobile devices, and media servers.

### Branding and credit screens
Branding overlays, watermarks, sponsor logos, channel intro animations, and outro credit rolls are out of scope. The tool is intended for personal and home use rather than commercial karaoke channels.

### Distribution automation
Automated publishing to YouTube (uploading video, generating descriptions, uploading thumbnails), syncing to cloud storage (Google Drive, Dropbox, Rclone), and dispatching Discord completion webhooks are out of scope. Users can upload or transfer rendered MP4 files manually.

### Medley generation
Chaining multiple songs together with automated crossfades and continuous subtitle streams is out of scope. Users who want medleys can assemble tracks in an audio editor or DAW before feeding the combined audio to the pipeline.

## Out of scope for now

### Automated audio download and search
Automatic searching and downloading of source audio from YouTube or FlacFetch by artist and song title is out of scope for now. The pipeline accepts an existing, user-provided audio file on disk. This avoids maintaining scrapers, rate-limit workarounds, and download tooling.

### Interactive review web GUI
A browser-based interactive GUI (such as Next.js/FastAPI) for adjusting word alignment markers, re-timing syllables, and correcting text typos is out of scope. 

Instead, review and manual corrections happen directly in workspace files (`lyrics.txt`, `lyrics_ruby.txt`, `guide.mid`, and `karaoke.ass`). Upstream and downstream dependencies are automatically managed via SHA-256 fingerprints in `manifest.json`.

Known limitation: editing start and end timestamps in `alignment.json` or ASS event lines by hand is significantly harder than dragging markers across an audio waveform. However, for Japanese lyrics, Whisper alignment plus mora-proportional splitting generally gets close enough that only ruby readings (`lyrics_ruby.txt`) or source text (`lyrics.txt`) require manual edits.

### Cloud execution and distributed workers
Offloading separation, transcription, or video rendering to remote cloud services (Google Cloud Run, GPU spot instances, distributed encoding fleets) is out of scope for now. The entire pipeline runs locally on Windows with PyTorch and FFmpeg. We may revisit remote rendering or separation only if local GPU hardware limits become a practical barrier.

## Out of scope pending investigation

### LLM-assisted lyric correction
Automated LLM agents that compare transcribed lyrics against web reference lyrics to fix hallucinated lines, misheard syllables, and line breaks are out of scope pending investigation. For Japanese vocals, Whisper transcription from `vocals.wav` frequently struggles with kanji selection, homophones, and English loan words. Providing a known `lyrics.txt` file is currently the reliable path. Automated alignment against reference lyrics without relying on external cloud LLM APIs remains an area to investigate.
