"""Vocab utility for the tokenized-spectrum encoder (alberts_baseline_tokenize.py).

`.build()` mirrors OpenNMT-py's onmt_build_vocab defaults -- what
alberts_baseline/benchmark/start_training.py calls to turn whitespace-tokenized
src files into a vocabulary: frequency-rank tokens, cap at `max_size` (OpenNMT
default src_vocab_size=32768), special tokens first, pad to a multiple of
`vocab_size_multiple` (OpenNMT default 8).
"""
import json
from collections import Counter
from pathlib import Path
from typing import List


class MMSpectrumVocab:
    UNK, PAD, BOS, EOS = "<unk>", "<blank>", "<s>", "</s>"

    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.unk_id = token_to_id[self.UNK]
        self.pad_id = token_to_id[self.PAD]
        self.bos_id = token_to_id[self.BOS]
        self.eos_id = token_to_id[self.EOS]

    def __len__(self):
        return len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(tok, self.unk_id) for tok in text.split()]

    def save(self, path):
        Path(path).write_text(json.dumps(self.token_to_id))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def build(cls, source_strings, max_size=32768, min_frequency=0, vocab_size_multiple=8):
        counts = Counter()
        for text in source_strings:
            counts.update(text.split())

        specials = [cls.UNK, cls.PAD, cls.BOS, cls.EOS]
        ranked = [tok for tok, freq in counts.most_common() if freq >= min_frequency]

        budget = max(max_size - len(specials), 0) if max_size else len(ranked)
        vocab_tokens = specials + ranked[:budget]

        if vocab_size_multiple > 1 and len(vocab_tokens) % vocab_size_multiple != 0:
            target_size = ((len(vocab_tokens) // vocab_size_multiple) + 1) * vocab_size_multiple
            for i in range(target_size - len(vocab_tokens)):
                vocab_tokens.append(f"<pad_extra_{i}>")

        token_to_id = {tok: i for i, tok in enumerate(vocab_tokens)}
        return cls(token_to_id)
