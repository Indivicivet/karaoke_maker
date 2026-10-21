# Setup (Windows)

## 1. System tools
| Tool | Why | Install |
|---|---|---|
| Python 3.11+ (64-bit) | Runtime | `winget install Python.Python.3.13` |
| FFmpeg with libass | Decoding, mixing, subtitle burn-in, encoding | `winget install Gyan.FFmpeg` |

Check that FFmpeg includes libass: `ffmpeg -hide_banner -filters | findstr " ass "` should print a line.

Python 3.14: `diffq-fixed` (pulled in by `audio-separator`) has no prebuilt Windows wheel for 3.14, so installing it needs Visual Studio Build Tools. `audio-separator` also refuses 3.14.1. Use 3.11–3.13 unless you already have the build tools.

## 2. Python environment
Install into your global Python or a venv.

**Why global install:** a CUDA build of torch is about 4–5 GB installed (see Disk space). A venv would hold its own copy of that, duplicating what most GPU machines already have globally.

**Why a venv:** the packages carry tight pins that can clash with other projects in the same Python:
- `torch`/`torchaudio` must stay at exactly 2.11.0 (see below).
- `audio-separator` pins `samplerate==0.1.0`, `beartype<0.19` and `rotary-embedding-torch<0.7`.
- `stable-ts` pins `openai-whisper<=20250625`.

If another project later upgrades any of these, this pipeline may break. In that case, re-run step 4 with the torch pins included. If conflicts become a real problem, `python -m venv --system-site-packages .venv` is a middle ground: it reuses the global torch and isolates only the other packages (~1 GB).

## 3. PyTorch (install first, pinned)
torchaudio's newest release is 2.11.0 and `stable-ts`/`torchcrepe` require it, so install torch and torchaudio as a matching pair. Pick one:

```powershell
# NVIDIA GPU: cu126, cu128 or cu130, whichever your driver supports
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# CPU only (works, but separation and Whisper are much slower)
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

## 4. Python packages
Repeat the torch pins in this command so pip cannot upgrade or replace the CUDA build:

```powershell
python -m pip install "torch==2.11.0" "torchaudio==2.11.0" "audio-separator[gpu]" stable-ts torchcrepe mido "fugashi[unidic-lite]" fonttools numpy soundfile
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

## 5. Disk space
| Item | Download | Installed (estimated) |
|---|---|---|
| torch 2.11.0 CUDA wheel | 1.8–2.6 GB (cu130 smallest) | ~4–5 GB |
| torch 2.11.0 CPU wheel | ~110 MB | ~0.5 GB |
| `onnxruntime-gpu` | ~260 MB | ~0.5 GB |
| Everything else | ~0.3 GB | ~0.5–1 GB |
| **Total (global, or a venv)** | | **~5–7 GB (GPU), ~1–1.5 GB (CPU)** |
| `cache/` (Whisper large-v3 ~3 GB, separation model, font, background) | first run | ~3.5–4 GB |

Download sizes are the real wheel file sizes (cp313 Windows, checked on PyPI and download.pytorch.org). Installed sizes are estimates.

## 6. First run
Models, the subtitle font and the default background are downloaded or generated on first use and stored in `cache/` inside the repo. The first run therefore takes longer and needs internet access. Song workspaces go to `songs/`.

```powershell
python karaoke_maker.py "song.mp3"
```
