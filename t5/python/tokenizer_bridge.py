import json
import sys

from transformers import AutoConfig, AutoTokenizer


model_name = sys.argv[1]
op = sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    legacy=False,
    model_max_length=512,
)
config = AutoConfig.from_pretrained(model_name)

if op == "ids":
    print(
        json.dumps(
            {
                "eos_id": tokenizer.eos_token_id,
                "decoder_start_id": config.decoder_start_token_id,
            }
        )
    )
elif op == "encode":
    out = tokenizer(
        sys.argv[3],
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"].squeeze(0).tolist()
    print(json.dumps(out))
elif op == "tokens":
    toks = json.loads(sys.argv[3])
    print(json.dumps(tokenizer.convert_ids_to_tokens(toks)))
elif op == "decode":
    payload = json.loads(sys.argv[3])
    toks = payload["tokens"]
    skip = payload.get("skip_special_tokens", True)
    print(json.dumps(tokenizer.decode(toks, skip_special_tokens=skip)))
else:
    raise ValueError(f"unsupported op: {op}")
