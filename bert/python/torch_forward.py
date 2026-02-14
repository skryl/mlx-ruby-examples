import json
import sys
from transformers import AutoModel, AutoTokenizer

model_name = sys.argv[1]
texts = json.loads(sys.argv[2])
tokenizer = AutoTokenizer.from_pretrained(model_name)
torch_model = AutoModel.from_pretrained(model_name)
torch_tokens = tokenizer(texts, return_tensors="pt", padding=True)
torch_forward = torch_model(**torch_tokens)

output = torch_forward.last_hidden_state.detach().cpu().numpy()
pooled = torch_forward.pooler_output
payload = {"last_hidden_state": output.tolist()}
if pooled is not None:
    payload["pooler_output"] = pooled.detach().cpu().numpy().tolist()
print(json.dumps(payload))
