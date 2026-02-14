# Whisper (MLX Ruby + DSL)

Ruby/DSL port of `mlx-examples/whisper` with a package-style layout and CLI.

## Files

- `whisper.rb`: core Whisper model classes (`ModelDimensions`, encoder/decoder, attention blocks).
- `audio.rb`: audio loading, padding, and lightweight log-mel feature extraction.
- `tokenizer.rb`: tokenizer/language token utilities.
- `decoding.rb`: decoding options/results plus greedy decode and language detection.
- `transcribe.rb`: high-level transcription API.
- `load_models.rb`: local/HF model loading.
- `writers.rb`: output writers (`txt`, `vtt`, `srt`, `tsv`, `json`, `all`).
- `timing.rb`: word-timestamp helper stubs.
- `cli.rb`: command-line interface.
- `convert.rb`: checkpoint conversion helper.
- `benchmark.rb`: lightweight benchmark script.
- `mlx_whisper.rb`: package entrypoint.
- `python/snapshot_download.py`: Hugging Face snapshot bridge.
- `python/extract_torch_weights.py`: torch checkpoint extraction bridge.

## Setup

Install optional Python dependencies for bridges/conversion:

```bash
pip install -r whisper/requirements.txt
```

Install FFmpeg to enable audio loading from files:

```bash
brew install ffmpeg
```

## Run CLI

```bash
ruby whisper/cli.rb --model mlx-community/whisper-tiny --output-format txt path/to/audio.wav
```

## API

```ruby
require_relative "whisper/mlx_whisper"

result = WhisperExample.transcribe("path/to/audio.wav", path_or_hf_repo: "mlx-community/whisper-tiny")
puts result["text"]
```

## Convert

```bash
ruby whisper/convert.rb --torch-name-or-path tiny --mlx-path whisper/mlx_models/tiny
```

## Test

```bash
ruby whisper/test.rb
```
