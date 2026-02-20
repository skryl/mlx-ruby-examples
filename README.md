# MLX Ruby Examples

[![CI](https://github.com/skryl/mlx-ruby-examples/actions/workflows/ci.yml/badge.svg)](https://github.com/skryl/mlx-ruby-examples/actions/workflows/ci.yml)

Ruby/MLX example ports that depend on Apple Metal when building the `mlx` gem.

## Prerequisites (macOS)

Install Xcode command line tools and make sure Xcode is selected:

```bash
xcode-select --install
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

Install the Metal Toolchain component required by `mlx` native build:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Verify Metal compiler tools are available:

```bash
xcrun -f metal
xcrun metal -v | head -n 1
```

## Install Ruby dependencies

Run this after the Metal setup above:

```bash
bundle install
```

## Benchmarks

Run model benchmarks against Python equivalents from the `mlx-examples` submodule:

```bash
bundle exec rake benchmark
```

Run benchmarks against the mirrored `no_dsl` Ruby versions:

```bash
bundle exec rake benchmark:no_dsl
```

The benchmark runner will:
- verify `mlx-examples` submodule files are present before running
- create/reuse a local venv at `tmp/benchmark/.venv`
- install Python benchmark dependencies into that venv
- run all Python model benchmarks first, then Ruby benchmarks
- write logs and JSON reports under `tmp/benchmark/`

Optional controls:

```bash
MODELS=mnist,bert RUNS=3 WARMUP=1 BENCH_TIMEOUT=1200 bundle exec rake benchmark
```

Recommended full run:

```bash
RUNS=10 WARMUP=2 bundle exec rake benchmark
```

Current DSL performance snapshot (2026-02-16, `RUNS=10 WARMUP=2`):

| model | py_cpu_avg_s | py_gpu_avg_s | rb_cpu_avg_s | rb_gpu_avg_s | rb/py_cpu | rb/py_gpu |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bert | 0.000917 | 0.002548 | 0.002470 | 0.003325 | 2.69x | 1.30x |
| cifar | 0.007630 | 0.004999 | 0.0108 | 0.006946 | 1.42x | 1.39x |
| clip | 0.000531 | 0.001213 | 0.000899 | 0.002167 | 1.69x | 1.79x |
| cvae | 0.003971 | 0.002777 | 0.006066 | 0.003660 | 1.53x | 1.32x |
| encodec | 0.001227 | 0.002710 | 0.002459 | 0.007535 | 2.00x | 2.78x |
| flux | 0.4856 | 0.4891 | 0.2684 | 0.0373 | 0.55x | 0.08x |
| gcn | 0.000479 | 0.001138 | 0.002739 | 0.003796 | 5.72x | 3.34x |
| llava | 0.005465 | 0.007193 | 0.007020 | 0.008629 | 1.28x | 1.20x |
| llms/gguf_llm | 0.001191 | 0.001970 | 0.002012 | 0.002300 | 1.69x | 1.17x |
| llms/llama | 0.000793 | 0.001407 | 0.001693 | 0.002142 | 2.13x | 1.52x |
| llms/mistral | 0.002519 | 0.003281 | 0.003875 | 0.002466 | 1.54x | 0.75x |
| llms/mixtral | 0.003524 | 0.005302 | 0.006508 | 0.008319 | 1.85x | 1.57x |
| llms/speculative_decoding | 0.001469 | 0.002831 | 0.003344 | 0.004229 | 2.28x | 1.49x |
| lora | 0.000126 | 0.000458 | 0.000253 | 0.000942 | 2.01x | 2.05x |
| mnist | 0.000410 | 0.000908 | 0.001376 | 0.001783 | 3.35x | 1.96x |
| musicgen | 0.001494 | 0.002660 | 0.004533 | 0.005715 | 3.03x | 2.15x |
| normalizing_flow | 0.000705 | 0.001749 | 0.002426 | 0.004664 | 3.44x | 2.67x |
| segment_anything | 0.0141 | 0.0143 | 0.0200 | 0.0221 | 1.41x | 1.55x |
| speechcommands | 0.002243 | 0.003062 | 0.003739 | 0.005169 | 1.67x | 1.69x |
| stable_diffusion | 0.000471 | 0.000833 | 0.001200 | 0.001719 | 2.55x | 2.06x |
| t5 | 0.002911 | 0.004580 | 0.005373 | 0.007287 | 1.85x | 1.59x |
| transformer_lm | 0.000928 | 0.001720 | 0.005840 | 0.007065 | 6.29x | 4.11x |
| whisper | 0.005921 | 0.006771 | 0.009091 | 0.0166 | 1.54x | 2.45x |

All models reported parity matches for input/output shape and content on CPU and GPU (`✓/✓` in benchmark output).
