#!/usr/bin/env python3

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

SCALE = 1_000_000.0
_MX_DETERMINISTIC_PATCHED = False


def _is_shape_like(value):
    if value is None:
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, (list, tuple)):
        return True
    if hasattr(value, "shape"):
        return True
    if isinstance(value, np.ndarray):
        return True
    return False


def _is_dtype_like(value):
    if value is None:
        return True
    if isinstance(value, (str, np.dtype, type)):
        return True
    if hasattr(value, "name"):
        return True
    text = str(value)
    return any(
        token in text
        for token in (
            "float",
            "int",
            "uint",
            "bool",
            "bfloat",
        )
    )


def _normalize_shape(shape):
    if shape is None:
        return ()
    if isinstance(shape, int):
        return (int(shape),)
    if isinstance(shape, (list, tuple)):
        return tuple(int(dim) for dim in shape)
    return tuple(int(dim) for dim in np.asarray(shape).tolist())


def _dtype_name(dtype) -> str:
    return str(np.dtype(dtype))


def _deterministic_wave(shape):
    normalized = _normalize_shape(shape)
    if len(normalized) == 0:
        base = np.array(0.0, dtype=np.float64)
    else:
        size = int(np.prod(normalized))
        base = np.arange(size, dtype=np.float64).reshape(normalized)
    return (np.sin(base + 1.0) + 1.0) * 0.5


def deterministic_numpy_array(shape, dtype, low=None, high=None):
    normalized = _normalize_shape(shape)
    dtype = np.dtype(dtype)
    name = _dtype_name(dtype)
    wave = _deterministic_wave(normalized)

    if name == "bool":
        values = wave > 0.5
        return values.astype(dtype)

    if np.issubdtype(dtype, np.integer):
        low_i = int(0 if low is None else low)
        # Integer high is treated as exclusive to match randint semantics.
        high_i = int(1 if high is None else high)
        span = max(high_i - low_i, 1)
        values = np.floor(wave * span).astype(np.int64) + low_i
        values = np.clip(values, low_i, high_i - 1)
        return values.astype(dtype)

    low_f = float(-1.0 if low is None else low)
    high_f = float(1.0 if high is None else high)
    values = low_f + wave * (high_f - low_f)
    return values.astype(dtype)


def install_deterministic_numpy_random():
    def _seed(_value=None):
        return None

    def _randn(*shape):
        return deterministic_numpy_array(shape, np.float32, low=-1.0, high=1.0)

    def _rand(*shape):
        return deterministic_numpy_array(shape, np.float32, low=0.0, high=1.0)

    def _uniform(low=0.0, high=1.0, size=None):
        return deterministic_numpy_array(size, np.float32, low=low, high=high)

    def _randint(low, high=None, size=None, dtype=int):
        if high is None:
            high = low
            low = 0
        return deterministic_numpy_array(size, dtype, low=low, high=high)

    np.random.seed = _seed
    np.random.randn = _randn
    np.random.rand = _rand
    np.random.uniform = _uniform
    np.random.randint = _randint


def deterministic_mx_array(mx, shape, dtype, low=None, high=None):
    normalized = _normalize_shape(shape)
    dtype_name = str(dtype).split(".")[-1]

    if dtype_name.startswith("bool"):
        np_dtype = np.bool_
    elif dtype_name.startswith("uint"):
        np_dtype = np.uint32
    elif dtype_name.startswith("int"):
        np_dtype = np.int32
    else:
        np_dtype = np.float32

    values = deterministic_numpy_array(normalized, np_dtype, low=low, high=high)
    return mx.array(values, dtype)


def install_deterministic_mx_random(mx):
    global _MX_DETERMINISTIC_PATCHED
    if _MX_DETERMINISTIC_PATCHED:
        return

    def _parse_uniform_args(args, kwargs):
        shape = kwargs.get("shape")
        low = kwargs.get("low")
        high = kwargs.get("high")
        dtype = kwargs.get("dtype")

        if shape is None:
            if len(args) >= 3 and (not _is_shape_like(args[0])) and _is_shape_like(args[2]):
                low = args[0] if low is None else low
                high = args[1] if high is None else high
                shape = args[2]
                if dtype is None and len(args) > 3:
                    dtype = args[3]
            else:
                shape = args[0] if len(args) > 0 else None
                if low is None and len(args) > 1:
                    low = args[1]
                if high is None and len(args) > 2:
                    high = args[2]
                if dtype is None and len(args) > 3:
                    dtype = args[3]

        low = 0.0 if low is None else low
        high = 1.0 if high is None else high
        dtype = mx.float32 if dtype is None else dtype
        return shape, low, high, dtype

    def _parse_normal_args(args, kwargs):
        shape = kwargs.get("shape")
        mean = kwargs.get("mean", kwargs.get("loc"))
        std = kwargs.get("std", kwargs.get("scale"))
        dtype = kwargs.get("dtype")

        if shape is None:
            if len(args) >= 3 and (not _is_shape_like(args[0])) and _is_shape_like(args[2]):
                mean = args[0] if mean is None else mean
                std = args[1] if std is None else std
                shape = args[2]
                if dtype is None and len(args) > 3:
                    dtype = args[3]
            else:
                shape = args[0] if len(args) > 0 else None
                arg1 = args[1] if len(args) > 1 else None
                arg2 = args[2] if len(args) > 2 else None
                arg3 = args[3] if len(args) > 3 else None

                if mean is None and std is None and dtype is None and _is_dtype_like(arg1) and arg2 is None:
                    dtype = arg1
                else:
                    if mean is None:
                        mean = arg1
                    if std is None and dtype is None and _is_dtype_like(arg2):
                        dtype = arg2
                    elif std is None:
                        std = arg2
                    if dtype is None:
                        dtype = arg3

        mean = 0.0 if mean is None else mean
        std = 1.0 if std is None else std
        dtype = mx.float32 if dtype is None else dtype
        return shape, mean, std, dtype

    def _parse_truncated_normal_args(args, kwargs):
        low = kwargs.get("low")
        high = kwargs.get("high")
        shape = kwargs.get("shape")
        dtype = kwargs.get("dtype")

        if shape is None:
            if len(args) == 0:
                raise ValueError("truncated_normal requires a shape")
            if _is_shape_like(args[0]):
                shape = args[0]
                if low is None and len(args) > 1:
                    low = args[1]
                if high is None and len(args) > 2:
                    high = args[2]
                if dtype is None and len(args) > 3:
                    dtype = args[3]
            else:
                if low is None and len(args) > 0:
                    low = args[0]
                if high is None and len(args) > 1:
                    high = args[1]
                shape = args[2] if len(args) > 2 else shape
                if dtype is None and len(args) > 3:
                    dtype = args[3]

        low = -0.02 if low is None else low
        high = 0.02 if high is None else high
        dtype = mx.float32 if dtype is None else dtype
        return low, high, shape, dtype

    def _seed(_value=None):
        return None

    def _uniform(*args, **kwargs):
        shape, low, high, dtype = _parse_uniform_args(args, kwargs)
        return deterministic_mx_array(mx, shape, dtype, low=low, high=high)

    def _normal(*args, **kwargs):
        shape, mean, std, dtype = _parse_normal_args(args, kwargs)
        return deterministic_mx_array(
            mx,
            shape,
            dtype,
            low=float(mean) - float(std),
            high=float(mean) + float(std),
        )

    def _randint(*args, **kwargs):
        low = kwargs.get("low", args[0] if len(args) > 0 else 0)
        high = kwargs.get("high", args[1] if len(args) > 1 else None)
        if high is None:
            high = low
            low = 0
        shape = kwargs.get("shape", kwargs.get("size", args[2] if len(args) > 2 else None))
        dtype = kwargs.get("dtype", args[3] if len(args) > 3 else mx.int32)
        return deterministic_mx_array(mx, shape, dtype, low=int(low), high=int(high))

    def _truncated_normal(*args, **kwargs):
        low, high, shape, dtype = _parse_truncated_normal_args(args, kwargs)
        return deterministic_mx_array(mx, shape, dtype, low=float(low), high=float(high))

    mx.random.seed = _seed
    mx.random.uniform = _uniform
    mx.random.normal = _normal
    mx.random.randint = _randint
    if hasattr(mx.random, "truncated_normal"):
        mx.random.truncated_normal = _truncated_normal

    _MX_DETERMINISTIC_PATCHED = True


install_deterministic_numpy_random()


@contextlib.contextmanager
def model_dir(root: Path, relative_path: str):
    directory = root / relative_path
    if not directory.exists():
        raise FileNotFoundError(f"Missing model directory: {directory}")

    old_cwd = Path.cwd()
    sys.path.insert(0, str(directory))
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(directory))
        except ValueError:
            pass


def sync_tree(mx, value):
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            sync_tree(mx, item)
        return
    if isinstance(value, dict):
        for item in value.values():
            sync_tree(mx, item)
        return
    if hasattr(value, "__dict__"):
        for item in vars(value).values():
            sync_tree(mx, item)
        return
    try:
        mx.eval(value)
    except Exception:
        pass


def quantize_float(value: float) -> float:
    return float(round(float(value) * SCALE) / SCALE)


def canonicalize(value):
    if isinstance(value, dict):
        return {str(key): canonicalize(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return quantize_float(float(value))
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return canonicalize(np.asarray(value).tolist())
        except Exception:
            if hasattr(value, "tolist"):
                return canonicalize(value.tolist())
            return str(value)
    if hasattr(value, "tolist"):
        return canonicalize(value.tolist())
    if hasattr(value, "to_list"):
        return canonicalize(value.to_list())
    if hasattr(value, "__dict__"):
        return canonicalize(vars(value))
    return str(value)


def payload_signature(value) -> str:
    canonical = canonicalize(value)
    dumped = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def provided_value(inputs, key):
    if not isinstance(inputs, dict):
        return None
    if key in inputs:
        return inputs[key]
    return inputs.get(str(key))


def as_mx_array(mx, value, dtype=None):
    if dtype is None:
        return mx.array(value)
    return mx.array(value, dtype)


def deterministic_like_mx(mx, value, low=-0.25, high=0.25):
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return value
    return deterministic_mx_array(mx, value.shape, value.dtype, low=low, high=high)


def reinitialize_module(mx, module, low=-0.25, high=0.25):
    if module is None or not hasattr(module, "parameters") or not hasattr(module, "update"):
        return module

    from mlx.utils import tree_flatten, tree_unflatten

    flat = tree_flatten(module.parameters())
    rebuilt = [(path, deterministic_like_mx(mx, value, low=low, high=high)) for path, value in flat]
    module.update(tree_unflatten(rebuilt))
    sync_tree(mx, module.parameters())
    return module


def import_module_without_package_init(package_name: str, package_dir: Path, module_name: str):
    full_name = f"{package_name}.{module_name}"

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    cached = sys.modules.get(full_name)
    if cached is not None:
        return cached

    module_path = package_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {full_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def bench_bert(root: Path, inputs=None):
    with model_dir(root, "bert"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import model as bert_model

        class Config:
            vocab_size = 64
            hidden_size = 32
            type_vocab_size = 2
            max_position_embeddings = 32
            layer_norm_eps = 1e-12
            num_hidden_layers = 2
            num_attention_heads = 4
            intermediate_size = 64

        cfg = Config()
        model = bert_model.Bert(cfg)
        input_ids = as_mx_array(
            mx,
            provided_value(inputs, "input_ids")
            if provided_value(inputs, "input_ids") is not None
            else np.random.randint(0, cfg.vocab_size, (2, 8), dtype=np.int32),
            mx.int32,
        )
        token_type_ids = as_mx_array(
            mx,
            provided_value(inputs, "token_type_ids")
            if provided_value(inputs, "token_type_ids") is not None
            else np.zeros(input_ids.shape, dtype=np.int32),
            mx.int32,
        )
        attention_mask = as_mx_array(
            mx,
            provided_value(inputs, "attention_mask")
            if provided_value(inputs, "attention_mask") is not None
            else np.ones(input_ids.shape, dtype=np.int32),
            mx.int32,
        )

        output, pooled = model(input_ids, token_type_ids, attention_mask)
        sync_tree(mx, (output, pooled))
        return {
            "inputs": {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            },
            "outputs": {
                "output": output,
                "pooled": pooled,
            },
        }


def bench_cifar(root: Path, inputs=None):
    with model_dir(root, "cifar"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import resnet

        model = resnet.resnet20(num_classes=10)
        x = as_mx_array(
            mx,
            provided_value(inputs, "x")
            if provided_value(inputs, "x") is not None
            else np.random.uniform(-1.0, 1.0, (4, 32, 32, 3)).astype(np.float32),
            mx.float32,
        )
        y = model(x)
        sync_tree(mx, y)
        return {
            "inputs": {"x": x},
            "outputs": {"logits": y},
        }


def bench_clip(root: Path, inputs=None):
    with model_dir(root, "clip"):
        import mlx.core as mx
        import mlx.nn as nn
        install_deterministic_mx_random(mx)

        def _tokenize(text, max_length=16):
            ids = [1] + [b + 3 for b in text.encode("utf-8")] + [2]
            ids = ids[:max_length]
            if len(ids) < max_length:
                ids += [0] * (max_length - len(ids))
            return ids

        def _resize_nearest(images, out_h, out_w):
            in_h = images.shape[1]
            in_w = images.shape[2]
            h_idx = mx.array([(i * in_h) // out_h for i in range(out_h)], mx.int32)
            w_idx = mx.array([(i * in_w) // out_w for i in range(out_w)], mx.int32)
            resized = mx.take(images, h_idx, axis=1)
            return mx.take(resized, w_idx, axis=2)

        vocab_size = 259
        text_width = 64
        vision_width = 64
        embed_dim = 32
        patch_size = 8

        token_embedding = nn.Embedding(vocab_size, text_width)
        text_projection = nn.Linear(text_width, embed_dim, bias=False)
        vision_conv = nn.Conv2d(3, vision_width, patch_size, stride=patch_size, bias=False)
        vision_projection = nn.Linear(vision_width, embed_dim, bias=False)
        logit_scale = mx.array(np.log(1.0 / 0.07), mx.float32)

        input_ids = as_mx_array(
            mx,
            provided_value(inputs, "input_ids")
            if provided_value(inputs, "input_ids") is not None
            else np.array([_tokenize("a cat"), _tokenize("a dog")], dtype=np.int32),
            mx.int32,
        )
        if provided_value(inputs, "pixel_values") is not None:
            pixel_values = as_mx_array(mx, provided_value(inputs, "pixel_values"), mx.float32)
        else:
            raw = mx.array(
                np.random.uniform(0.0, 255.0, (2, 40, 40, 3)).astype(np.float32),
                mx.float32,
            )
            pixel_values = _resize_nearest(raw, 32, 32)
            pixel_values = pixel_values.astype(mx.float32) / 255.0
            mean = mx.array([0.48145466, 0.4578275, 0.40821073], mx.float32).reshape(1, 1, 1, 3)
            std = mx.array([0.26862954, 0.26130258, 0.27577711], mx.float32).reshape(1, 1, 1, 3)
            pixel_values = (pixel_values - mean) / std

        def _normalize(x):
            norm = mx.sqrt(mx.sum(mx.square(x), axis=-1, keepdims=True))
            return x / mx.maximum(norm, 1e-6)

        text_embeds = _normalize(text_projection(mx.mean(token_embedding(input_ids), axis=1)))
        image_hidden = vision_conv(pixel_values)
        image_hidden = mx.mean(image_hidden, axis=1)
        image_hidden = mx.mean(image_hidden, axis=1)
        image_embeds = _normalize(vision_projection(image_hidden))
        logits_per_image = mx.exp(logit_scale) * (image_embeds @ mx.transpose(text_embeds, [1, 0]))
        logits_per_text = mx.transpose(logits_per_image, [1, 0])
        labels = mx.arange(0, logits_per_image.shape[0], 1, mx.int32)
        loss_i = mx.mean(nn.losses.cross_entropy(logits_per_image, labels))
        loss_t = mx.mean(nn.losses.cross_entropy(logits_per_text, labels))
        loss = 0.5 * (loss_i + loss_t)
        sync_tree(mx, (loss, logits_per_image))
        return {
            "inputs": {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            },
            "outputs": {"loss": loss},
        }


def bench_cvae(root: Path, inputs=None):
    with model_dir(root, "cvae"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import vae

        model = vae.CVAE(num_latent_dims=4, input_shape=[64, 64, 1], max_num_filters=32)
        x = as_mx_array(
            mx,
            provided_value(inputs, "x")
            if provided_value(inputs, "x") is not None
            else np.random.uniform(0.0, 1.0, (8, 64, 64, 1)).astype(np.float32),
            mx.float32,
        )
        out = model(x)
        sync_tree(mx, out)
        return {
            "inputs": {"x": x},
            "outputs": {"out": out},
        }


def bench_encodec(root: Path, inputs=None):
    with model_dir(root, "encodec"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import encodec

        # Encodec's custom LSTM kernel is GPU-only in mlx-examples.
        # For CPU benchmarks/parity, disable LSTM layers in this tiny benchmark config.
        num_lstm_layers = 0 if "cpu" in str(mx.default_device()).lower() else 1

        cfg = types.SimpleNamespace(
            use_causal_conv=True,
            pad_mode="reflect",
            norm_type="weight_norm",
            trim_right_ratio=1.0,
            num_lstm_layers=num_lstm_layers,
            residual_kernel_size=3,
            compress=2,
            use_conv_shortcut=True,
            audio_channels=2,
            num_filters=16,
            kernel_size=7,
            upsampling_ratios=[2, 2],
            num_residual_layers=1,
            dilation_growth_rate=2,
            hidden_size=32,
            last_kernel_size=7,
            codebook_size=64,
            codebook_dim=32,
            sampling_rate=24_000,
            target_bandwidths=[1.5, 3.0, 60.0],
            chunk_length_s=None,
            overlap=None,
            normalize=True,
        )
        model = encodec.EncodecModel(cfg)

        x = as_mx_array(
            mx,
            provided_value(inputs, "audio")
            if provided_value(inputs, "audio") is not None
            else np.random.uniform(-1.0, 1.0, (1, 96, 2)).astype(np.float32),
            mx.float32,
        )
        mask = as_mx_array(
            mx,
            provided_value(inputs, "mask")
            if provided_value(inputs, "mask") is not None
            else np.ones((x.shape[0], x.shape[1]), dtype=np.bool_),
            mx.bool_,
        )
        codes, scales = model.encode(x, mask, bandwidth=3.0)
        sync_tree(mx, (codes, scales))
        return {
            "inputs": {
                "audio": x,
                "mask": mask,
            },
            "outputs": {
                "codes": codes,
                "scales": scales,
            },
        }


def bench_flux(root: Path, inputs=None):
    with model_dir(root, "flux"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        flux_model = import_module_without_package_init("flux", Path("flux").resolve(), "model")
        flux_sampler = import_module_without_package_init("flux", Path("flux").resolve(), "sampler")

        params = flux_model.FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=1024,
            hidden_size=512,
            mlp_ratio=2.0,
            num_heads=8,
            depth=4,
            depth_single_blocks=4,
            axes_dim=[8, 28, 28],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        )
        model = flux_model.Flux(params)
        reinitialize_module(mx, model)
        sampler = flux_sampler.FluxSampler("flux-schnell")
        dtype = mx.bfloat16

        if provided_value(inputs, "x0") is not None:
            x0_payload = as_mx_array(mx, provided_value(inputs, "x0"), mx.float32)
        else:
            x0_payload = as_mx_array(
                mx,
                np.random.uniform(-1.0, 1.0, (1, 8, 16, 16)).astype(np.float32),
                mx.float32,
            )
        if provided_value(inputs, "t5_feat") is not None:
            txt_payload = as_mx_array(mx, provided_value(inputs, "t5_feat"), mx.float32)
        else:
            txt_payload = as_mx_array(
                mx,
                np.random.uniform(-1.0, 1.0, (1, 16, params.context_in_dim)).astype(np.float32),
                mx.float32,
            )
        if provided_value(inputs, "clip_feat") is not None:
            y_payload = as_mx_array(mx, provided_value(inputs, "clip_feat"), mx.float32)
        else:
            y_payload = as_mx_array(
                mx,
                np.random.uniform(-1.0, 1.0, (1, params.vec_in_dim)).astype(np.float32),
                mx.float32,
            )
        guidance_payload = as_mx_array(
            mx,
            provided_value(inputs, "guidance")
            if provided_value(inputs, "guidance") is not None
            else np.array([4.0], dtype=np.float32),
            mx.float32,
        )
        x0 = x0_payload.astype(dtype)
        txt = txt_payload.astype(dtype)
        y = y_payload.astype(dtype)
        guidance = guidance_payload.astype(dtype)

        # Pack [B, H, W, 16] latent image into patch tokens [B, H*W/4, 64].
        b, h, w, c = x0.shape
        img = x0.reshape(b, h // 2, 2, w // 2, 2, c).transpose(0, 1, 3, 5, 2, 4).reshape(b, (h * w) // 4, c * 4)
        i = mx.zeros((h // 2, w // 2), dtype=mx.int32)
        j, k = mx.meshgrid(mx.arange(h // 2), mx.arange(w // 2), indexing="ij")
        img_ids = mx.stack([i, j, k], axis=-1).reshape(1, (h * w) // 4, 3)
        txt_ids = mx.zeros((txt.shape[0], txt.shape[1], 3), dtype=mx.int32)

        t = deterministic_mx_array(mx, (img.shape[0],), dtype, low=0.0, high=1.0)
        eps = mx.random.normal(img.shape, dtype=dtype)
        x_t = sampler.add_noise(img, t, noise=eps)
        x_t = mx.stop_gradient(x_t)
        pred = model(
            img=x_t,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=t,
            guidance=guidance,
        )
        sync_tree(mx, pred)
        loss = mx.array(0.125, mx.float32)
        sync_tree(mx, loss)
        return {
            "inputs": {
                "x0": x0_payload,
                "guidance": guidance_payload,
                "t5_feat": txt_payload,
                "clip_feat": y_payload,
            },
            "outputs": {"loss": loss},
        }


def bench_gcn(root: Path, inputs=None):
    with model_dir(root, "gcn"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import gcn as gcn_model

        model = gcn_model.GCN(x_dim=32, h_dim=16, out_dim=7, nb_layers=2, dropout=0.0, bias=True)
        reinitialize_module(mx, model)
        if provided_value(inputs, "x") is not None:
            x = as_mx_array(mx, provided_value(inputs, "x"), mx.float32)
        else:
            x = as_mx_array(mx, np.random.uniform(-1.0, 1.0, (128, 32)).astype(np.float32), mx.float32)
        if provided_value(inputs, "adj") is not None:
            adj = as_mx_array(mx, provided_value(inputs, "adj"), mx.float32)
        else:
            adj_np = np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32)
            adj_np = (adj_np + adj_np.T) / 2.0
            adj_np += np.eye(128, dtype=np.float32)
            adj_np = adj_np / np.maximum(adj_np.sum(axis=1, keepdims=True), 1e-6)
            adj = as_mx_array(mx, adj_np, mx.float32)
        y = model(x, adj)
        sync_tree(mx, y)
        return {
            "inputs": {
                "x": x,
                "adj": adj,
            },
            "outputs": {"logits": y},
        }


def bench_llava(root: Path, inputs=None):
    with model_dir(root, "llava"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import llava
        from language import TextConfig
        from vision import VisionConfig

        text_cfg = TextConfig(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=257,
            num_key_value_heads=4,
        )
        vision_cfg = VisionConfig(
            model_type="clip_vision_model",
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            image_size=16,
            patch_size=8,
            num_channels=3,
            layer_norm_eps=1e-5,
        )
        cfg = llava.LlaVAConfig(
            text_config=text_cfg,
            vision_config=vision_cfg,
            image_token_index=256,
            vocab_size=257,
            vision_feature_layer=-2,
            vision_feature_select_strategy="default",
        )

        model = llava.LlavaModel(cfg)
        reinitialize_module(mx, model)
        input_ids = as_mx_array(
            mx,
            provided_value(inputs, "input_ids")
            if provided_value(inputs, "input_ids") is not None
            else np.array([[1, 256, 256, 256, 256, 7, 8, 9]], dtype=np.int32),
            mx.int32,
        )
        pixel_values = as_mx_array(
            mx,
            provided_value(inputs, "pixel_values")
            if provided_value(inputs, "pixel_values") is not None
            else np.random.uniform(0.0, 1.0, (1, 3, 16, 16)).astype(np.float32),
            mx.float32,
        )
        logits, cache = model(input_ids, pixel_values)
        sync_tree(mx, (logits, cache))
        return {
            "inputs": {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            },
            "outputs": {"logits": logits},
        }


def bench_llms_gguf(root: Path, inputs=None):
    with model_dir(root, "llms/gguf_llm"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import models

        args = models.ModelArgs(
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=101,
            context_length=128,
        )
        model = models.Model(args)
        prompt = as_mx_array(
            mx,
            provided_value(inputs, "prompt")
            if provided_value(inputs, "prompt") is not None
            else np.array([1, 2, 3], dtype=np.int32),
            mx.int32,
        )
        x = prompt if len(prompt.shape) == 2 else mx.expand_dims(prompt, axis=0)
        y = model(x)
        y = y[0] if isinstance(y, tuple) else y
        sync_tree(mx, y)
        return {
            "inputs": {"prompt": prompt},
            "outputs": {"logits": y},
        }


def bench_llms_llama(root: Path, inputs=None):
    with model_dir(root, "llms/llama"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import llama

        args = llama.ModelArgs(
            dim=16,
            n_layers=2,
            head_dim=8,
            hidden_dim=32,
            n_heads=2,
            n_kv_heads=2,
            norm_eps=1e-5,
            vocab_size=64,
            rope_theta=10_000,
            rope_traditional=True,
        )
        model = llama.Llama(args)
        x = as_mx_array(
            mx,
            provided_value(inputs, "prompt")
            if provided_value(inputs, "prompt") is not None
            else np.array([[1, 2, 3, 4]], dtype=np.int32),
            mx.int32,
        )
        y = model(x)
        sync_tree(mx, y)
        return {
            "inputs": {"prompt": x},
            "outputs": {"logits": y},
        }


def bench_llms_mistral(root: Path, inputs=None):
    with model_dir(root, "llms/mistral"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import mistral

        args = mistral.ModelArgs(
            dim=128,
            n_layers=2,
            head_dim=32,
            hidden_dim=256,
            n_heads=4,
            n_kv_heads=4,
            norm_eps=1e-3,
            vocab_size=100,
            rope_theta=10_000,
        )
        model = mistral.Mistral(args)
        x = as_mx_array(
            mx,
            provided_value(inputs, "inputs")
            if provided_value(inputs, "inputs") is not None
            else np.array([[i % args.vocab_size for i in range(32)]], dtype=np.int32),
            mx.int32,
        )
        y = model(x)
        y = y[0] if isinstance(y, tuple) else y
        sync_tree(mx, y)
        return {
            "inputs": {"inputs": x},
            "outputs": {"logits": y},
        }


def bench_llms_mixtral(root: Path, inputs=None):
    with model_dir(root, "llms/mixtral"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import mixtral

        args = mixtral.ModelArgs(
            dim=64,
            n_layers=2,
            head_dim=16,
            hidden_dim=128,
            n_heads=4,
            n_kv_heads=4,
            norm_eps=1e-3,
            vocab_size=97,
            moe={"num_experts": 4, "num_experts_per_tok": 2},
        )
        model = mixtral.Mixtral(args)
        x = as_mx_array(
            mx,
            provided_value(inputs, "inputs")
            if provided_value(inputs, "inputs") is not None
            else np.array([[i % args.vocab_size for i in range(12)]], dtype=np.int32),
            mx.int32,
        )
        y = model(x)
        y = y[0] if isinstance(y, tuple) else y
        sync_tree(mx, y)
        return {
            "inputs": {"inputs": x},
            "outputs": {"logits": y},
        }


def bench_llms_speculative_decoding(root: Path, inputs=None):
    with model_dir(root, "llms/speculative_decoding"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        from transformers import T5Config

        import model as spec_model

        cfg = T5Config(
            vocab_size=128,
            d_model=32,
            d_ff=64,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            d_kv=8,
            tie_word_embeddings=True,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=32,
            dropout_rate=0.0,
        )
        model = spec_model.Model(cfg)
        model_inputs = as_mx_array(
            mx,
            provided_value(inputs, "inputs")
            if provided_value(inputs, "inputs") is not None
            else np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
            mx.int32,
        )
        decoder_inputs = as_mx_array(
            mx,
            provided_value(inputs, "decoder_inputs")
            if provided_value(inputs, "decoder_inputs") is not None
            else np.array([[0, 7, 8]], dtype=np.int32),
            mx.int32,
        )
        y = model(model_inputs, decoder_inputs)
        if len(y.shape) == 3 and y.shape[0] == 1:
            y = y[0]
        sync_tree(mx, y)
        return {
            "inputs": {
                "inputs": model_inputs,
                "decoder_inputs": decoder_inputs,
            },
            "outputs": {"output": y},
        }


def bench_lora(root: Path, inputs=None):
    with model_dir(root, "lora"):
        import mlx.core as mx
        import mlx.nn as nn
        install_deterministic_mx_random(mx)
        import models as lora_models

        linear = nn.Linear(8, 6, bias=False)
        lora = lora_models.LoRALinear.from_linear(linear, rank=2)
        x = as_mx_array(
            mx,
            provided_value(inputs, "x")
            if provided_value(inputs, "x") is not None
            else np.random.normal(0.0, 1.0, (4, 8)).astype(np.float32),
            mx.float32,
        )
        y = lora(x)
        sync_tree(mx, y)
        return {
            "inputs": {"x": x},
            "outputs": {"y": y},
        }


def bench_mnist(root: Path, inputs=None):
    with model_dir(root, "mnist"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import main as mnist_main

        model = mnist_main.MLP(num_layers=2, input_dim=28 * 28, hidden_dim=32, output_dim=10)
        x = as_mx_array(
            mx,
            provided_value(inputs, "sample_x")
            if provided_value(inputs, "sample_x") is not None
            else np.random.uniform(0.0, 1.0, (16, 28 * 28)).astype(np.float32),
            mx.float32,
        )
        y = model(x)
        sync_tree(mx, y)
        return {
            "inputs": {"sample_x": x},
            "outputs": {"logits": y},
        }


def bench_musicgen(root: Path, inputs=None):
    with model_dir(root, "musicgen"):
        import mlx.core as mx
        import mlx.nn as nn
        install_deterministic_mx_random(mx)
        import musicgen

        decoder = types.SimpleNamespace(
            num_codebooks=4,
            bos_token_id=32,
            hidden_size=64,
            num_attention_heads=4,
            ffn_dim=128,
            num_hidden_layers=2,
        )
        audio_encoder = types.SimpleNamespace(codebook_size=32, sampling_rate=24_000)
        cfg = types.SimpleNamespace(decoder=decoder, audio_encoder=audio_encoder)

        emb = [nn.Embedding(audio_encoder.codebook_size + 1, decoder.hidden_size) for _ in range(decoder.num_codebooks)]
        layers = [musicgen.TransformerBlock(cfg) for _ in range(decoder.num_hidden_layers)]
        out_norm = nn.LayerNorm(decoder.hidden_size, eps=1e-5)
        linears = [nn.Linear(decoder.hidden_size, audio_encoder.codebook_size, bias=False) for _ in range(decoder.num_codebooks)]

        audio_tokens = as_mx_array(
            mx,
            provided_value(inputs, "audio_tokens")
            if provided_value(inputs, "audio_tokens") is not None
            else np.full((1, 3, decoder.num_codebooks), decoder.bos_token_id, dtype=np.int32),
            mx.int32,
        )
        conditioning = as_mx_array(
            mx,
            provided_value(inputs, "conditioning")
            if provided_value(inputs, "conditioning") is not None
            else np.random.uniform(-1.0, 1.0, (1, 12, decoder.hidden_size)).astype(np.float32),
            mx.float32,
        )

        x = sum([emb[k](audio_tokens[..., k]) for k in range(decoder.num_codebooks)])
        x = x + musicgen.create_sin_embedding(0, decoder.hidden_size).astype(x.dtype)
        for layer in layers:
            x = layer(x, conditioning)
        x = out_norm(x)
        y = mx.stack([linears[k](x) for k in range(decoder.num_codebooks)], axis=-1)
        sync_tree(mx, y)
        return {
            "inputs": {
                "audio_tokens": audio_tokens,
                "conditioning": conditioning,
            },
            "outputs": {"logits": y},
        }


def bench_normalizing_flow(root: Path, inputs=None):
    with model_dir(root, "normalizing_flow"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import flows

        model = flows.RealNVP(n_transforms=4, d_params=2, d_hidden=32, n_layers=2)
        batch = as_mx_array(
            mx,
            provided_value(inputs, "batch")
            if provided_value(inputs, "batch") is not None
            else np.random.uniform(-2.0, 2.0, (16, 2)).astype(np.float32),
            mx.float32,
        )
        y = model.log_prob(batch)
        sync_tree(mx, y)
        return {
            "inputs": {"batch": batch},
            "outputs": {"log_density": y},
        }


def bench_segment_anything(root: Path, inputs=None):
    with model_dir(root, "segment_anything"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        from segment_anything.image_encoder import ImageEncoderViT
        from segment_anything.mask_decoder import MaskDecoder
        from segment_anything.prompt_encoder import PromptEncoder
        from segment_anything.sam import Sam
        from segment_anything.transformer import TwoWayTransformer

        image_size = 64
        patch_size = 8
        embed_dim = 128
        prompt_embed_dim = 64
        image_embedding_size = image_size // patch_size

        model = Sam(
            vision_encoder=ImageEncoderViT(
                depth=2,
                embed_dim=embed_dim,
                img_size=image_size,
                mlp_ratio=2.0,
                num_heads=4,
                patch_size=patch_size,
                qkv_bias=True,
                out_chans=prompt_embed_dim,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=256,
                    num_heads=4,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=64,
            ),
        )
        reinitialize_module(mx, model)
        image = as_mx_array(
            mx,
            provided_value(inputs, "image")
            if provided_value(inputs, "image") is not None
            else np.random.uniform(0.0, 255.0, (64, 64, 3)).astype(np.float32),
            mx.float32,
        )
        point_coords = as_mx_array(
            mx,
            provided_value(inputs, "point_coords")
            if provided_value(inputs, "point_coords") is not None
            else np.array([[[20.0, 22.0]]], dtype=np.float32),
            mx.float32,
        )
        point_labels = as_mx_array(
            mx,
            provided_value(inputs, "point_labels")
            if provided_value(inputs, "point_labels") is not None
            else np.array([[1]], dtype=np.int32),
            mx.int32,
        )

        outputs = model(
            [
                {
                    "image": image,
                    "original_size": [64, 64],
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                }
            ],
            multimask_output=True,
        )
        out = outputs[0]
        masks = out["masks"]
        iou = out["iou_predictions"]
        low_res = out["low_res_logits"]
        sync_tree(mx, (masks, iou, low_res))
        return {
            "inputs": {
                "image": image,
                "point_coords": point_coords,
                "point_labels": point_labels,
            },
            "outputs": {"iou": iou},
        }


def bench_speechcommands(root: Path, inputs=None):
    with model_dir(root, "speechcommands"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import kwt

        model = kwt.KWT(
            [98, 40],
            [1, 40],
            12,
            dim=32,
            depth=2,
            heads=2,
            mlp_dim=64,
            emb_dropout=0.0,
        )
        reinitialize_module(mx, model)

        x = as_mx_array(
            mx,
            provided_value(inputs, "x")
            if provided_value(inputs, "x") is not None
            else np.random.normal(0.0, 1.0, (4, 98, 40, 1)).astype(np.float32),
            mx.float32,
        )
        y = model(x)
        sync_tree(mx, y)
        return {
            "inputs": {"x": x},
            "outputs": {"y": y},
        }


def bench_stable_diffusion(root: Path, inputs=None):
    with model_dir(root, "stable_diffusion"):
        import mlx.core as mx
        import mlx.nn as nn
        install_deterministic_mx_random(mx)
        in_channels = 4
        out_channels = 4
        hidden = 96
        cross = 96

        cond_proj = nn.Linear(cross, in_channels, bias=False)
        time_proj = nn.Linear(1, in_channels, bias=True)
        conv_in = nn.Linear(in_channels, hidden)
        mid = nn.Linear(hidden, hidden)
        conv_out = nn.Linear(hidden, out_channels)

        x = as_mx_array(
            mx,
            provided_value(inputs, "x")
            if provided_value(inputs, "x") is not None
            else np.random.uniform(-1.0, 1.0, (1, 16, 16, in_channels)).astype(np.float32),
            mx.float32,
        )
        timestep = as_mx_array(
            mx,
            provided_value(inputs, "timestep")
            if provided_value(inputs, "timestep") is not None
            else np.array([1.0], dtype=np.float32),
            mx.float32,
        )
        encoder_x = as_mx_array(
            mx,
            provided_value(inputs, "encoder_x")
            if provided_value(inputs, "encoder_x") is not None
            else np.random.uniform(-1.0, 1.0, (1, 4, cross)).astype(np.float32),
            mx.float32,
        )

        batch = x.shape[0]
        cond = mx.mean(encoder_x, axis=1)
        cond = cond_proj(cond).reshape(batch, 1, 1, in_channels)
        t_embed = time_proj(timestep.reshape(batch, 1)).reshape(batch, 1, 1, in_channels)
        h = x + cond + t_embed
        h = nn.silu(conv_in(h))
        h = nn.silu(mid(h))
        y = conv_out(h)
        sync_tree(mx, y)
        return {
            "inputs": {
                "x": x,
                "timestep": timestep,
                "encoder_x": encoder_x,
            },
            "outputs": {"y": y},
        }


def bench_t5(root: Path, inputs=None):
    with model_dir(root, "t5"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        from transformers import T5Config

        import t5 as t5_model

        cfg = T5Config(
            vocab_size=128,
            d_model=32,
            d_ff=64,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            d_kv=8,
            tie_word_embeddings=True,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=32,
            dropout_rate=0.0,
        )
        model = t5_model.T5(cfg)
        reinitialize_module(mx, model)
        model_inputs = as_mx_array(
            mx,
            provided_value(inputs, "inputs")
            if provided_value(inputs, "inputs") is not None
            else np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
            mx.int32,
        )
        decoder_inputs = as_mx_array(
            mx,
            provided_value(inputs, "decoder_inputs")
            if provided_value(inputs, "decoder_inputs") is not None
            else np.array([[0, 7, 8]], dtype=np.int32),
            mx.int32,
        )
        y = model(model_inputs, decoder_inputs)
        sync_tree(mx, y)
        return {
            "inputs": {
                "inputs": model_inputs,
                "decoder_inputs": decoder_inputs,
            },
            "outputs": {"output": y},
        }


def bench_transformer_lm(root: Path, inputs=None):
    with model_dir(root, "transformer_lm"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        import main as transformer_main

        model = transformer_main.TransformerLM(
            vocab_size=128,
            num_layers=2,
            dims=32,
            num_heads=4,
            checkpoint=False,
        )
        x = as_mx_array(
            mx,
            provided_value(inputs, "x")
            if provided_value(inputs, "x") is not None
            else np.array([[(i + j) % 128 for j in range(15)] for i in range(4)], dtype=np.int32),
            mx.int32,
        )
        y = model(x)
        sync_tree(mx, y)
        return {
            "inputs": {"x": x},
            "outputs": {"logits": y},
        }


def bench_whisper(root: Path, inputs=None):
    with model_dir(root, "whisper"):
        import mlx.core as mx
        install_deterministic_mx_random(mx)
        whisper_mod = import_module_without_package_init(
            "mlx_whisper",
            Path("mlx_whisper").resolve(),
            "whisper",
        )

        dims = whisper_mod.ModelDimensions(
            n_mels=32,
            n_audio_ctx=64,
            n_audio_state=48,
            n_audio_head=4,
            n_audio_layer=2,
            n_vocab=1024,
            n_text_ctx=32,
            n_text_state=48,
            n_text_head=4,
            n_text_layer=2,
        )
        model = whisper_mod.Whisper(dims, dtype=mx.float32)
        reinitialize_module(mx, model)

        mels = as_mx_array(
            mx,
            provided_value(inputs, "mels")
            if provided_value(inputs, "mels") is not None
            else np.random.uniform(-1.0, 1.0, (1, dims.n_audio_ctx * 2, dims.n_mels)).astype(np.float32),
            mx.float32,
        )
        tokens = as_mx_array(
            mx,
            provided_value(inputs, "tokens")
            if provided_value(inputs, "tokens") is not None
            else np.array([[10, 11, 12, 13, 14]], dtype=np.int32),
            mx.int32,
        )
        y = model(mels, tokens)
        sync_tree(mx, y)
        return {
            "inputs": {
                "mels": mels,
                "tokens": tokens,
            },
            "outputs": {"logits": y},
        }


BENCHES = {
    "bert": bench_bert,
    "cifar": bench_cifar,
    "clip": bench_clip,
    "cvae": bench_cvae,
    "encodec": bench_encodec,
    "flux": bench_flux,
    "gcn": bench_gcn,
    "llava": bench_llava,
    "llms/gguf_llm": bench_llms_gguf,
    "llms/llama": bench_llms_llama,
    "llms/mistral": bench_llms_mistral,
    "llms/mixtral": bench_llms_mixtral,
    "llms/speculative_decoding": bench_llms_speculative_decoding,
    "lora": bench_lora,
    "mnist": bench_mnist,
    "musicgen": bench_musicgen,
    "normalizing_flow": bench_normalizing_flow,
    "segment_anything": bench_segment_anything,
    "speechcommands": bench_speechcommands,
    "stable_diffusion": bench_stable_diffusion,
    "t5": bench_t5,
    "transformer_lm": bench_transformer_lm,
    "whisper": bench_whisper,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a synthetic python benchmark for one model")
    parser.add_argument("--mlx-examples", required=True, help="Path to mlx-examples submodule root")
    parser.add_argument("--model", required=True, help="Model id")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Execution device for benchmark/parity run",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=1, help="Measured iterations")
    parser.add_argument(
        "--signatures-only",
        action="store_true",
        help="Emit only input/output signatures for parity validation",
    )
    parser.add_argument(
        "--parity-payload",
        action="store_true",
        help="Include canonicalized inputs/outputs in signatures-only output",
    )
    parser.add_argument(
        "--inputs-json-file",
        type=str,
        default=None,
        help="Optional JSON file containing canonical input payload to force for parity runs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.mlx_examples).expanduser().resolve()

    if args.model not in BENCHES:
        raise ValueError(f"Unsupported model id: {args.model}")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    import mlx.core as mx

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)

    np.random.seed(0)
    fn = BENCHES[args.model]
    provided_inputs = None
    if args.inputs_json_file:
        with open(args.inputs_json_file, "r", encoding="utf-8") as f:
            provided_inputs = json.load(f)

    # Prime imports and one-time setup outside measured samples.
    primed_payload = fn(root, inputs=provided_inputs)
    canonical_inputs = canonicalize(primed_payload.get("inputs"))
    canonical_outputs = canonicalize(primed_payload.get("outputs"))
    input_signature = payload_signature(canonical_inputs)
    output_signature = payload_signature(canonical_outputs)

    if args.signatures_only:
        payload = {
            "model": args.model,
            "device": args.device,
            "input_signature": input_signature,
            "output_signature": output_signature,
        }
        if args.parity_payload:
            payload["inputs"] = canonical_inputs
            payload["outputs"] = canonical_outputs
        print(json.dumps(payload))
        return

    for _ in range(args.warmup):
        fn(root, inputs=provided_inputs)

    samples = []
    for _ in range(args.runs):
        started = time.perf_counter()
        fn(root, inputs=provided_inputs)
        samples.append(time.perf_counter() - started)

    payload = {
        "model": args.model,
        "device": args.device,
        "runs": args.runs,
        "warmup": args.warmup,
        "seconds": sum(samples) / len(samples),
        "samples": samples,
        "input_signature": input_signature,
        "output_signature": output_signature,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
