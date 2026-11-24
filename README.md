# karaoke_maker

Local Windows CLI pipeline that turns an audio file into a Japanese arcade-style (DAM / JOYSOUND) karaoke MP4 with furigana wipe subtitles and vocal guide options.

See [SETUP.md](SETUP.md) for installation and [DESIGN.md](DESIGN.md) for full architecture and stage details.

## Usage

```powershell
# Default run (continuous guide melody at 0.3)
python karaoke_maker.py "song.mp3" -l "lyrics.txt"

# Softened/de-sung filtered vocals (no MIDI/alignment dependency)
python karaoke_maker.py "song.mp3" -l "lyrics.txt" --guide-vol 0 --filtered-vocal-vol 0.6
```

## Guide & vocal levers

Audio mixing exposes three independent guide levers on top of `--inst-vol` (default `1.0`):

- `--guide-vol` (default `0.3`): synthesized guide melody (`--guide-timbre`: `continuous`, `vibraphone`, `glockenspiel`, or `simple`).
- `--filtered-vocal-vol` (default `0.0`): softened, de-sung version of `vocals.wav` (`vocals_filtered.wav`) built purely from signal processing, independent of Whisper/alignment timing.
- `--vocal-vol` (default `0.0`, or `1.0` with `split-lr`): original isolated vocal stem.

Note on `--guide-timbre simple`: the simple guide mode is currently cursed (continuous raw-pitch synth tracks raw vocal slides/glitches a bit too literally). If you use it, it is probably best to keep `--guide-vol` around `0.2` layered under `--filtered-vocal-vol` around `0.6`:

```powershell
python karaoke_maker.py "song.mp3" -l "lyrics.txt" --guide-timbre simple --guide-vol 0.2 --filtered-vocal-vol 0.6
```
