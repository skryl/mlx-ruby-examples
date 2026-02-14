import argparse
import collections
import copy
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten


class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = args.head_dim ** -0.5
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(args.head_dim, traditional=args.rope_traditional, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.wq(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        k, v = map(repeat, (k, v))
        if cache is not None:
            kc, vc = cache
            q = self.rope(q, offset=kc.shape[2])
            k = self.rope(k, offset=kc.shape[2])
            k = mx.concatenate([kc, k], axis=2)
            v = mx.concatenate([vc, v], axis=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = (scores @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(out), (k, v)


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        r, c = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        return h + r, c


class Llama(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)


def sanitize_config(config, weights):
    config.pop("model_type", None)
    n_heads = config["n_heads"]
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads
    if "hidden_dim" not in config:
        config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
    if config.get("vocab_size", -1) < 0:
        config["vocab_size"] = weights["output.weight"].shape[-1]
    if "rope_theta" not in config:
        config["rope_theta"] = 10000
    if "rope_traditional" not in config:
        config["rope_traditional"] = True
    config.pop("multiple_of", None)
    config.pop("ffn_dim_multiplier", None)
    return config


def torch_to_mx(a: torch.Tensor, dtype: str):
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))


def llama(model_path, dtype: str):
    SHARD_FIRST = ["wv", "wq", "wk", "w1", "w3", "output"]
    SHARD_SECOND = ["tok_embeddings", "wo", "w2"]
    SHARD_WEIGHTS = set(SHARD_FIRST + SHARD_SECOND)

    def shard_key(k):
        keys = k.split(".")
        if len(keys) < 2:
            return None
        return keys[-2]

    def unshard(k, v):
        wn = shard_key(k)
        if wn not in SHARD_WEIGHTS:
            return v
        elif wn in SHARD_FIRST:
            axis = 0
        elif wn in SHARD_SECOND:
            axis = 1
        else:
            raise ValueError("Invalid weight name")
        return mx.concatenate(v, axis=axis)

    torch_files = glob.glob(str(model_path / "consolidated.*.pth"))
    weights = collections.defaultdict(list)
    for wf in torch_files:
        state = torch.load(wf, map_location=torch.device("cpu"))
        for k, v in state.items():
            v = torch_to_mx(v, dtype=dtype)
            state[k] = None
            if shard_key(k) in SHARD_WEIGHTS:
                weights[k].append(v)
            else:
                weights[k] = v

    for k, v in weights.items():
        weights[k] = unshard(k, v)
    with open(model_path / "params.json", "r") as f:
        params = json.loads(f.read())
    return weights, params


def tiny_llama(model_path, dtype: str):
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(str(model_path)).state_dict()
    config = transformers.AutoConfig.from_pretrained(model_path)
    model = {k.replace("model.", ""): v for k, v in model.items()}
    model = {k.replace("mlp", "feed_forward"): v for k, v in model.items()}
    model = {k.replace("down_proj", "w2"): v for k, v in model.items()}
    model = {k.replace("up_proj", "w3"): v for k, v in model.items()}
    model = {k.replace("gate_proj", "w1"): v for k, v in model.items()}
    model = {k.replace("input_layernorm", "attention_norm"): v for k, v in model.items()}
    model = {k.replace("post_attention_layernorm", "ffn_norm"): v for k, v in model.items()}
    model = {k.replace("lm_head", "output"): v for k, v in model.items()}
    model = {k.replace("embed_tokens", "tok_embeddings"): v for k, v in model.items()}
    model = {k.replace("self_attn", "attention"): v for k, v in model.items()}
    model = {k.replace("q_proj", "wq"): v for k, v in model.items()}
    model = {k.replace("k_proj", "wk"): v for k, v in model.items()}
    model = {k.replace("v_proj", "wv"): v for k, v in model.items()}
    model = {k.replace("o_proj", "wo"): v for k, v in model.items()}

    params = {
        "dim": config.hidden_size,
        "hidden_dim": config.intermediate_size,
        "n_heads": config.num_attention_heads,
        "n_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "norm_eps": config.rms_norm_eps,
        "rope_traditional": False,
    }
    if hasattr(config, "num_key_value_heads"):
        params["n_kv_heads"] = config.num_key_value_heads

    weights = {k: torch_to_mx(v, dtype=dtype) for k, v in model.items()}
    return weights, params


def quantize(weights, config, q_group_size, q_bits):
    quantized_config = copy.deepcopy(config)
    config = sanitize_config(config, weights)
    model = Llama(ModelArgs(**config))
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))
    nn.quantize(model, q_group_size, q_bits)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))
    return quantized_weights, quantized_config


def make_shards(weights, max_file_size_gibibyte=15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard = {}
    shard_size = 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
parser.add_argument("--torch-path", type=str, required=True)
parser.add_argument("--mlx-path", type=str, default="mlx_model")
parser.add_argument("--model-name", choices=["tiny_llama", "llama"], default="llama")
parser.add_argument("-q", "--quantize", action="store_true")
parser.add_argument("--q-group-size", type=int, default=64)
parser.add_argument("--q-bits", type=int, default=4)
parser.add_argument("--dtype", type=str, default="float16")
args = parser.parse_args()

torch_path = Path(args.torch_path)
mlx_path = Path(args.mlx_path)
mlx_path.mkdir(parents=True, exist_ok=True)

weights, params = globals()[args.model_name](torch_path, dtype=args.dtype)
params["model_type"] = "llama"
if args.quantize:
    weights, params = quantize(weights, params, args.q_group_size, args.q_bits)

shutil.copyfile(str(torch_path / "tokenizer.model"), str(mlx_path / "tokenizer.model"))
shards = make_shards(weights)
if len(shards) == 1:
    mx.savez(str(mlx_path / "weights.npz"), **shards[0])
else:
    for i, shard in enumerate(shards):
        mx.savez(str(mlx_path / f"weights.{i:02d}.npz"), **shard)
with open(mlx_path / "config.json", "w") as fid:
    json.dump(params, fid, indent=4)
