# EnCodec (MLX Ruby + DSL)

Ruby/DSL port of `mlx-examples/encodec`.

## Files

- `encodec.rb`: EnCodec model/config, processor, quantizer, and pretrained-loading helpers.
- `utils.rb`: audio I/O utilities (`save_audio`, `load_audio`) with FFmpeg-based decoding.
- `example.rb`: encode/decode demo script.
- `convert.rb`: checkpoint conversion utility.
- `test.rb`: local model/preprocess/quantizer/audio utility tests.
- `python/snapshot_download.py`: Hugging Face snapshot bridge used by Ruby scripts.
- `python/upload_to_hub.py`: optional Hugging Face upload bridge for converted checkpoints.

## Setup

Install optional Python dependencies used by bridge scripts:

```bash
pip install -r encodec/requirements.txt
```

Install FFmpeg to enable `load_audio`:

```bash
brew install ffmpeg
```

## Convert Checkpoint

```bash
ruby encodec/convert.rb --model 48khz --dtype float32 --output encodec/mlx_models/encodec
```

## Run Example

With synthetic audio:

```bash
ruby encodec/example.rb --seconds 1.0 --output encodec/reconstructed.wav
```

With an input audio file:

```bash
ruby encodec/example.rb --input path/to/audio.wav --output encodec/reconstructed.wav --bandwidth 3.0
```

## Test

```bash
ruby encodec/test.rb
```
