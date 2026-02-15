# codex-examples

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
