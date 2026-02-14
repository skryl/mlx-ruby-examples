import json
import sys
import sentencepiece.sentencepiece_model_pb2 as model
import tempfile

payload = json.loads(sys.argv[1])
tokens = payload["tokens"]
bos = int(payload["bos"])
eos = int(payload["eos"])
unk = int(payload["unk"])
scores = payload.get("scores")
token_types = payload.get("token_types")

normalizer_spec = model.NormalizerSpec(
    name="identity",
    precompiled_charsmap=b"",
    add_dummy_prefix=True,
    remove_extra_whitespaces=False,
    normalization_rule_tsv=b"",
)
trainer_spec = model.TrainerSpec(
    model_type="BPE",
    vocab_size=len(tokens),
    input_format="text",
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    treat_whitespace_as_suffix=False,
    split_digits=True,
    allow_whitespace_only_pieces=True,
    vocabulary_output_piece_score=True,
    byte_fallback=True,
    unk_id=unk,
    bos_id=bos,
    eos_id=eos,
    pad_id=-1,
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    pad_piece="<pad>",
    pretokenization_delimiter="",
)
m = model.ModelProto(trainer_spec=trainer_spec, normalizer_spec=normalizer_spec)

for i, token in enumerate(tokens):
    score = scores[i] if scores else 0
    token_type = token_types[i] if token_types else 0
    m.pieces.append(
        model.ModelProto.SentencePiece(piece=token, score=float(score), type=int(token_type))
    )

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".model")
tmp.write(m.SerializeToString())
tmp.close()
print(tmp.name)
