# GGUF LLM (MLX Ruby + DSL)

Ruby/DSL port of `mlx-examples/llms/gguf_llm`.

## Files

- `models.rb`: GGUF model loader, tokenizer bridge, and generation core.
- `utils.rb`: sentencepiece tokenizer construction helpers from GGUF metadata.
- `generate.rb`: CLI text generation entrypoint.
- `test.rb`: local tests for model behavior and helper mappings.

## Setup

Install Python dependencies used by tokenizer and download bridges:

```bash
pip install sentencepiece protobuf==3.20.2 huggingface_hub
```

## Run

Using a local GGUF file:

```bash
ruby llms/gguf_llm/generate.rb \
  --gguf /path/to/model.gguf \
  --prompt "Write a quicksort in Python" \
  --max-tokens 100
```

Download from Hugging Face repo automatically:

```bash
ruby llms/gguf_llm/generate.rb \
  --repo TheBloke/Mistral-7B-v0.1-GGUF \
  --gguf mistral-7b-v0.1.Q8_0.gguf \
  --prompt "Write a quicksort in Python"
```

## Notes

- Supported quantization paths in this port match the Python example behavior:
  - `Q4_0` / `Q4_1` (4-bit)
  - `Q8_0` (8-bit)
- Other GGUF quantizations are loaded with a warning and cast to `float16`.

## Test

```bash
ruby llms/gguf_llm/test.rb
```

