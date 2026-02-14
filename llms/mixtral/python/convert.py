import argparse
import copy
import glob
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
    moe: dict


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
        self.rope = nn.RoPE(args.head_dim, traditional=True, base=1000000)

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


class MOEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.moe["num_experts"]
        self.num_experts_per_tok = args.moe["num_experts_per_tok"]
        self.experts = [FeedForward(args) for _ in range(self.num_experts)]
        self.gate = nn.Linear(args.dim, self.num_experts, bias=False)


class MOETransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MOEFeedForward(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)


class Mixtral(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [MOETransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)


def convert(tf, config):
    def convert_single(k, v):
        v = v.to(torch.float16).numpy()
        if "block_sparse_moe" not in k:
            return [(k, v)]
        if "gate" in k:
            return [(k.replace("block_sparse_moe", "feed_forward"), v)]

        num_experts = config["moe"]["num_experts"]
        key_path = k.split(".")
        v = np.split(v, num_experts, axis=0)
        if key_path[-1] == "w2":
            v = [u.T for u in v]

        w_name = key_path.pop()
        key_path[-1] = "feed_forward.experts"
        return [
            (".".join(key_path + [str(e), w_name, "weight"]), u)
            for e, u in enumerate(v)
        ]

    state = torch.load(tf, map_location=torch.device("cpu"))
    weights = {}
    for k, v in state.items():
        weights.update(convert_single(k, v))
    return weights


def quantize(weights, config, q_group_size, q_bits):
    quantized_config = copy.deepcopy(config)
    config = copy.deepcopy(config)
    config.pop("quantization", None)

    model = Mixtral(ModelArgs(**config))
    all_weights = dict(tree_flatten(model.parameters()))

    weights = tree_map(mx.array, weights)
    all_weights.update(weights)
    all_weights = tree_unflatten(list(all_weights.items()))
    model.update(all_weights)

    nn.quantize(model, q_group_size, q_bits)

    all_weights = dict(tree_flatten(model.parameters()))
    quantized_weights = {}
    for k, v in all_weights.items():
        if k not in weights:
            continue
        quantized_weights[k] = v
        prefix = k.split(".")[:-1]
        for qw in ["scales", "biases"]:
            qk = ".".join(prefix + [qw])
            if qk in all_weights:
                quantized_weights[qk] = all_weights[qk]

    quantized_config["quantization"] = {
        "group_size": q_group_size,
        "bits": q_bits,
    }
    return quantized_weights, quantized_config


parser = argparse.ArgumentParser(description="Convert Mixtral weights to MLX.")
parser.add_argument("--torch-path", type=str, default="Mixtral-8x7B-v0.1")
parser.add_argument("--mlx-path", type=str, default="mlx_model")
parser.add_argument("--params-path", type=str, required=True)
parser.add_argument("-q", "--quantize", action="store_true")
parser.add_argument("--q-group-size", type=int, default=64)
parser.add_argument("--q-bits", type=int, default=4)
args = parser.parse_args()

torch_path = Path(args.torch_path)
mlx_path = Path(args.mlx_path)
mlx_path.mkdir(parents=True, exist_ok=True)

with open(args.params_path, "r") as fid:
    config = json.load(fid)

shutil.copyfile(str(torch_path / "tokenizer.model"), str(mlx_path / "tokenizer.model"))

torch_files = glob.glob(str(torch_path / "consolidated.*.pt"))
torch_files = sorted(torch_files, key=lambda tf: int(tf.split(".")[-2]))
for e, tf in enumerate(torch_files):
    print(f"[INFO] Converting file {e + 1}/{len(torch_files)}")
    weights = convert(tf, config)
    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args.q_group_size, args.q_bits)
    np.savez(str(mlx_path / f"weights.{e}.npz"), **weights)

config["model_type"] = "mixtral"
with open(mlx_path / "config.json", "w") as f:
    json.dump(config, f, indent=4)
