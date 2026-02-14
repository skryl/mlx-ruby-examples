import argparse
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5EncoderModel


def embed(model_name: str):
    batch = [
        "translate English to German: That is good.",
        "This is an example of T5 working on MLX.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_model = T5EncoderModel.from_pretrained(model_name)
    torch_tokens = tokenizer(batch, return_tensors="pt", padding=True)
    torch_forward = torch_model(**torch_tokens, output_hidden_states=True)
    torch_output = torch_forward.last_hidden_state.detach().numpy().tolist()
    print(json.dumps({"batch": batch, "embedding": torch_output}))


def generate(model_name: str):
    prompt = (
        "translate English to German: As much as six inches of rain could fall in "
        "the New York City region through Monday morning, and officials warned of "
        "flooding along the coast."
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    torch_tokens = tokenizer(prompt, return_tensors="pt", padding=True).input_ids
    outputs = torch_model.generate(torch_tokens, do_sample=False, max_length=512)
    print(json.dumps({"output": tokenizer.decode(outputs[0], skip_special_tokens=True)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace T5 utility bridge")
    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--encode-only", action="store_true", default=False)
    args = parser.parse_args()
    if args.encode_only:
      embed(args.model)
    else:
      generate(args.model)
