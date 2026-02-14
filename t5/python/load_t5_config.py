import json
import sys
from transformers import T5Config

cfg = T5Config.from_pretrained(sys.argv[1])
keys = [
    "d_model",
    "d_kv",
    "d_ff",
    "num_heads",
    "num_layers",
    "num_decoder_layers",
    "layer_norm_epsilon",
    "relative_attention_num_buckets",
    "relative_attention_max_distance",
    "feed_forward_proj",
    "tie_word_embeddings",
    "vocab_size",
    "decoder_start_token_id",
]
print(json.dumps({k: getattr(cfg, k) for k in keys}))
