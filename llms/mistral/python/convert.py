import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float = 10000


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.scale = args.head_dim**-0.5
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(args.head_dim, traditional=True, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        b, l, _ = x.shape
        queries = self.wq(x).reshape(b, l, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.wk(x).reshape(b, l, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.wv(x).reshape(b, l, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.wo(out), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        residual, cache = self.attention(self.attention_norm(x), mask, cache)
        hidden = x + residual
        residual = self.feed_forward(self.ffn_norm(hidden))
        return hidden + residual, cache


class Mistral(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)


def quantize(weights, config, q_group_size, q_bits):
    quantized_config = copy.deepcopy(config)
    config = copy.deepcopy(config)
    config.pop("sliding_window", None)

    model = Mistral(ModelArgs(**config))
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))
    nn.quantize(model, q_group_size, q_bits)

    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))
    return quantized_weights, quantized_config


parser = argparse.ArgumentParser(description="Convert Mistral weights to MLX.")
parser.add_argument("--torch-path", type=str, default="mistral-7B-v0.1")
parser.add_argument("--mlx-path", type=str, default="mlx_model")
parser.add_argument("-q", "--quantize", action="store_true")
parser.add_argument("--q-group-size", type=int, default=64)
parser.add_argument("--q-bits", type=int, default=4)
args = parser.parse_args()

torch_path = Path(args.torch_path)
state = torch.load(str(torch_path / "consolidated.00.pth"), map_location=torch.device("cpu"))
mlx_path = Path(args.mlx_path)
mlx_path.mkdir(parents=True, exist_ok=True)

weights = {k: v.to(torch.float16).numpy() for k, v in state.items()}
with open(torch_path / "params.json", "r") as f:
    config = json.loads(f.read())

if args.quantize:
    weights, config = quantize(weights, config, args.q_group_size, args.q_bits)

np.savez(str(mlx_path / "weights.npz"), **weights)
shutil.copyfile(str(torch_path / "tokenizer.model"), str(mlx_path / "tokenizer.model"))

config["model_type"] = "mistral"
with open(mlx_path / "config.json", "w") as f:
    json.dump(config, f, indent=4)
