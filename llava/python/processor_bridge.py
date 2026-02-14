#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import requests
from PIL import Image
from transformers import AutoProcessor


def load_image(image_source: str):
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw)

    image_path = Path(image_source)
    if not image_path.is_file():
        raise ValueError(f"image must be a valid URL or existing file: {image_source}")
    return Image.open(image_path)


def main():
    if len(sys.argv) < 4:
        raise SystemExit("usage: processor_bridge.py <model_path> <op> <payload_json>")

    model_path = sys.argv[1]
    op = sys.argv[2]
    payload = json.loads(sys.argv[3]) if sys.argv[3] else {}

    tokenizer_config = payload.get("tokenizer_config", {})
    processor = AutoProcessor.from_pretrained(model_path, **tokenizer_config)

    if op == "meta":
        print(json.dumps({"eos_token_id": processor.tokenizer.eos_token_id}))
        return

    if op == "prepare":
        image = load_image(payload["image"])
        prompt = payload["prompt"]
        values = processor(text=prompt, images=image, return_tensors="np")
        print(
            json.dumps(
                {
                    "pixel_values": values["pixel_values"].tolist(),
                    "input_ids": values["input_ids"].tolist(),
                    "eos_token_id": processor.tokenizer.eos_token_id,
                }
            )
        )
        return

    if op == "decode":
        text = processor.tokenizer.decode(payload["tokens"])
        print(json.dumps(text))
        return

    raise ValueError(f"unknown op: {op}")


if __name__ == "__main__":
    main()
