import json
import sys
from transformers import AutoTokenizer

model_name = sys.argv[1]
texts = json.loads(sys.argv[2])
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoded = tokenizer(texts, return_tensors="np", padding=True)
out = {key: value.tolist() for key, value in encoded.items()}
print(json.dumps(out))
