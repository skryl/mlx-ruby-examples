import json
import sys

from transformers import AutoTokenizer

model_name = sys.argv[1]
op = sys.argv[2]
add_eos = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=add_eos)

if op == "eos":
    print(json.dumps(tokenizer.eos_token_id))
elif op == "encode":
    ids = tokenizer.encode(sys.argv[4])
    print(json.dumps(ids))
elif op == "decode":
    toks = json.loads(sys.argv[4])
    print(json.dumps(tokenizer.decode(toks)))
else:
    raise ValueError(f"unsupported op: {op}")
