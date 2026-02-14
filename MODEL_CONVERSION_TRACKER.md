# MLX Examples Conversion Tracker

## Scope

Track phased conversion of `mlx-examples` models to `mlx-ruby` using the same approach:

1. Convert all Python files to Ruby equivalents (including utilities).
2. Keep Python interop scripts in `<example>/python/*.py`.
3. Add `test.rb` and require passing tests before moving forward.
4. Add or update `README.md` with Ruby-first commands.

## Current Status

Already converted:

- `bert`
- `cifar`
- `llms` (`llama`, `mistral`, `mixtral`, `speculative_decoding`, `gguf_llm`)
- `mnist`
- `cvae`
- `transformer_lm`
- `normalizing_flow`
- `gcn`
- `speechcommands`
- `t5`
- `lora`
- `clip`
- `llava`
- `segment_anything`
- `stable_diffusion`
- `flux`
- `encodec`
- `musicgen`
- `whisper`

Pending:

- (none)

## Phase Tracker

| Phase | Scope | Status | Notes |
|---|---|---|---|
| 0 | Common conversion framework and guardrails | `pending` | |
| 1 | Low-complexity baselines (`mnist`, `cvae`, `transformer_lm`) | `completed` | |
| 2 | Data/classical examples (`speechcommands`, `gcn`, `normalizing_flow`) | `completed` | |
| 3 | HF mid-complexity (`t5`, `lora`, `clip`) | `completed` | |
| 4 | Multimodal perception (`llava`, `segment_anything`) | `completed` | |
| 5 | Generative image pipelines (`stable_diffusion`, `flux`) | `completed` | |
| 6 | Audio models (`whisper`, `encodec`, `musicgen`) | `completed` | |

## Model Checklist

| Model | Phase | Status | Tests Passing | README | Notes |
|---|---|---|---|---|---|
| `mnist` | 1 | `completed` | `yes` | `yes` | Python dataset bridge in `mnist/python/prepare_dataset.py`; synthetic fallback supported |
| `cvae` | 1 | `completed` | `yes` | `yes` | Saves reconstructions/samples as `.pgm`; supports synthetic data |
| `transformer_lm` | 1 | `completed` | `yes` | `yes` | Includes synthetic-token mode for offline smoke tests |
| `speechcommands` | 2 | `completed` | `yes` | `yes` | Synthetic MFSC-style dataset mode plus optional preprocessed NPZ loading |
| `gcn` | 2 | `completed` | `yes` | `yes` | Real Cora preprocessing via `gcn/python/prepare_cora.py`; synthetic default for fast tests |
| `normalizing_flow` | 2 | `completed` | `yes` | `yes` | Saves sampled arrays to `.npz` (`transform_*`, `original`) |
| `t5` | 3 | `completed` | `yes` | `yes` | Includes tokenizer/config/convert HF bridges in `t5/python/` |
| `lora` | 3 | `completed` | `yes` | `yes` | Synthetic base-model mode + LoRA adapter fuse workflow |
| `clip` | 3 | `completed` | `yes` | `yes` | Includes Ruby CLIP core + wrappers for upstream HF conversion/preproc utilities |
| `llava` | 4 | `completed` | `yes` | `yes` | Includes HF processor/snapshot bridges in `llava/python/` |
| `segment_anything` | 4 | `completed` | `yes` | `yes` | Includes Ruby SAM core modules + conversion bridge in `segment_anything/python/convert.py` |
| `stable_diffusion` | 5 | `completed` | `yes` | `yes` | Lightweight Ruby SD/SDXL pipeline with text-to-image and image-to-image CLIs |
| `flux` | 5 | `completed` | `yes` | `yes` | Includes Ruby FLUX pipeline/training stack + HF download bridge in `flux/python/hf_download.py` |
| `whisper` | 6 | `completed` | `yes` | `yes` | Includes Ruby package-style Whisper port (model/audio/tokenizer/decoding/transcribe/CLI/writers/convert) + Python bridge scripts |
| `encodec` | 6 | `completed` | `yes` | `yes` | Includes Ruby EnCodec model/config/processor + audio I/O utilities and HF bridge scripts |
| `musicgen` | 6 | `completed` | `yes` | `yes` | Includes Ruby MusicGen model/generation stack with EnCodec/T5 wrappers and Python weight bridges |

## Update Rule

Do not move to the next model until the current model’s `test.rb` passes.
