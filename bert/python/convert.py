import json
import sys
from pathlib import Path
import numpy
from transformers import AutoConfig, AutoModel


def replace_key(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".ln1.")
    key = key.replace(".output.LayerNorm.", ".ln2.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    key = key.replace(".LayerNorm.", ".norm.")
    key = key.replace("pooler.dense.", "pooler.")
    return key


model_name = sys.argv[1]
output_path = Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)

model = AutoModel.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

tensors = {
    replace_key(key): tensor.detach().cpu().numpy()
    for key, tensor in model.state_dict().items()
}
numpy.savez(str(output_path), **tensors)

config_path = output_path.with_suffix(".config.json")
config_path.write_text(config.to_json_string(), encoding="utf-8")

print(json.dumps({"weights_path": str(output_path), "config_path": str(config_path)}))
