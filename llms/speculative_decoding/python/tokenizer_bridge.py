import json
import sys
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(
    sys.argv[1],
    legacy=False,
    model_max_length=512,
)
op = sys.argv[2]

if op == "eos":
    print(json.dumps(tokenizer.eos_token_id))
elif op == "encode":
    output = tokenizer(
        sys.argv[3],
        return_tensors="np",
        return_attention_mask=False,
    )["input_ids"].squeeze(0).tolist()
    print(json.dumps(output))
elif op == "decode":
    toks = json.loads(sys.argv[3])
    print(json.dumps(tokenizer.decode(toks)))
else:
    raise ValueError(f"unsupported op: {op}")
