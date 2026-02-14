import json
import sys
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
op = sys.argv[2]

if op == "ids":
    print(json.dumps({"bos_id": sp.bos_id(), "eos_id": sp.eos_id(), "pad_id": sp.pad_id()}))
elif op == "encode":
    print(json.dumps(sp.encode(sys.argv[3])))
elif op == "decode":
    toks = json.loads(sys.argv[3])
    print(json.dumps(sp.decode(toks)))
elif op == "piece":
    print(json.dumps(sp.id_to_piece(int(sys.argv[3]))))
else:
    raise ValueError(f"unsupported op: {op}")
