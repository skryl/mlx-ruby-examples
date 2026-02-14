import json
import sys
from transformers import AutoConfig

required = [
    "vocab_size",
    "hidden_size",
    "type_vocab_size",
    "max_position_embeddings",
    "layer_norm_eps",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
]

cfg = AutoConfig.from_pretrained(sys.argv[1])
out = {key: getattr(cfg, key) for key in required}
print(json.dumps(out))
