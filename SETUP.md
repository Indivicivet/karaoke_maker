# Setup (Windows)

## 1. System tools
| Tool | Why | Install |
|---|---|---|
| Python 3.11 or 3.12 (64-bit) | Runtime | `winget install Python.Python.3.12` |
| FFmpeg with libass | Decoding, mixing, subtitle burn-in, encoding | `winget install Gyan.FFmpeg` |

Check that FFmpeg includes libass: `ffmpeg -hide_banner -filters | findstr " ass "` should print a line.

## 2. Python environment
Run these from the repo folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. PyTorch (install first)
Pick the command for your CUDA version from <https://pytorch.org/get-started/locally/>. Examples:

```powershell
# NVIDIA GPU (example: CUDA 12.6 build)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# CPU only (works, but separation and Whisper are much slower)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4. Python packages
```powershell
pip install "audio-separator[gpu]" stable-ts torchcrepe mido "fugashi[unidic-lite]" fonttools numpy soundfile
```
On a CPU-only machine, use `audio-separator[cpu]` instead.

| Package | Used for |
|---|---|
| `audio-separator` | Vocal / instrumental separation |
| `stable-ts` | Whisper transcription and lyric alignment |
| `torchcrepe` | Vocal pitch tracking for the guide melody |
| `mido` | Guide melody MIDI file |
| `fugashi[unidic-lite]` | Japanese tokenizing and kanji readings |
| `fonttools` | Font metrics for furigana and wipe layout |
| `numpy`, `soundfile` | Audio processing, chime synthesis, background frames |

## 5. First run
Models, the subtitle font and the default background are downloaded or generated on first use and stored in `cache/` inside the repo. The first run therefore takes longer and needs internet access. Song workspaces go to `songs/`.

```powershell
python karaoke_maker.py "song.mp3"
```
