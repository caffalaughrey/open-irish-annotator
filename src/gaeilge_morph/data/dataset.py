from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from torch.utils.data import Dataset, DataLoader


PAD_WORD_ID = 0
UNK_WORD_ID = 1
PAD_CHAR_ID = 0
EOS_CHAR_ID = 1
UNK_CHAR_ID = 2
BOS_CHAR_ID = 3


@dataclass
class EncodedSample:
    word_ids: torch.Tensor  # [T]
    char_ids: torch.Tensor  # [T, C]
    tag_ids: torch.Tensor  # [T]
    lemma_char_ids: torch.Tensor  # [T, L]


def encode_sentence(
    tokens: List[str],
    lemmas: List[str],
    tag_strings: List[str],
    word2id: Dict[str, int],
    char2id: Dict[str, int],
    tag2id: Dict[str, int],
    max_chars: int = 24,
    max_lemma: int = 24,
) -> EncodedSample:
    word_ids = [word2id.get(w, UNK_WORD_ID) for w in tokens]
    tag_ids = [tag2id[tag] for tag in tag_strings]
    # chars per token
    char_rows: List[List[int]] = []
    lemma_rows: List[List[int]] = []
    for w, lemma in zip(tokens, lemmas):
        ch_ids = [char2id.get(c, UNK_CHAR_ID) for c in w][: max_chars - 1]
        ch_ids = ch_ids + [PAD_CHAR_ID] * (max_chars - len(ch_ids))
        char_rows.append(ch_ids)

        # Targets do NOT include BOS; only true lemma chars followed by EOS
        le_ids = [char2id.get(c, UNK_CHAR_ID) for c in lemma][: max_lemma - 1]
        le_ids = le_ids + [EOS_CHAR_ID]
        le_ids = le_ids + [PAD_CHAR_ID] * (max_lemma - len(le_ids))
        lemma_rows.append(le_ids)

    return EncodedSample(
        word_ids=torch.tensor(word_ids, dtype=torch.long),
        char_ids=torch.tensor(char_rows, dtype=torch.long),
        tag_ids=torch.tensor(tag_ids, dtype=torch.long),
        lemma_char_ids=torch.tensor(lemma_rows, dtype=torch.long),
    )


class JSONLSentenceDataset(Dataset[EncodedSample]):
    def __init__(
        self,
        jsonl_path: Path,
        word2id: Dict[str, int],
        char2id: Dict[str, int],
        tag2id: Dict[str, int],
        max_chars: int = 24,
        max_lemma: int = 24,
    ) -> None:
        self.samples: List[EncodedSample] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                enc = encode_sentence(
                    obj["tokens"],
                    obj["lemmas"],
                    obj["tags"],
                    word2id,
                    char2id,
                    tag2id,
                    max_chars,
                    max_lemma,
                )
                self.samples.append(enc)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EncodedSample:
        return self.samples[idx]


def collate_batch(batch: List[EncodedSample]) -> Tuple[torch.Tensor, ...]:
    # Compute max lengths
    max_t = max(s.word_ids.shape[0] for s in batch)
    char_len = batch[0].char_ids.shape[1]
    lemma_len = batch[0].lemma_char_ids.shape[1]

    bsz = len(batch)
    word_ids = torch.full((bsz, max_t), PAD_WORD_ID, dtype=torch.long)
    tag_ids = torch.full((bsz, max_t), -100, dtype=torch.long)  # ignore_index for CE
    char_ids = torch.full((bsz, max_t, char_len), PAD_CHAR_ID, dtype=torch.long)
    lemma_char_ids = torch.full((bsz, max_t, lemma_len), PAD_CHAR_ID, dtype=torch.long)
    token_mask = torch.zeros((bsz, max_t), dtype=torch.bool)

    for i, s in enumerate(batch):
        t = s.word_ids.shape[0]
        word_ids[i, :t] = s.word_ids
        tag_ids[i, :t] = s.tag_ids
        char_ids[i, :t] = s.char_ids
        lemma_char_ids[i, :t] = s.lemma_char_ids
        token_mask[i, :t] = 1

    return word_ids, char_ids, tag_ids, lemma_char_ids, token_mask


def make_loader(
    dataset: JSONLSentenceDataset,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch, num_workers=2, pin_memory=False)


def make_length_bucketed_loader(
    dataset: JSONLSentenceDataset,
    batch_size: int = 16,
    buckets: int = 10,
    shuffle: bool = True,
) -> DataLoader:
    # Simple length bucketing: sort indices by sentence length and batch contiguously
    lengths = [int(s.word_ids.shape[0]) for s in dataset.samples]
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: List[List[int]] = []
    for i in range(0, len(sorted_indices), batch_size):
        batches.append(sorted_indices[i : i + batch_size])

    class _BucketSampler(torch.utils.data.Sampler[List[int]]):
        def __init__(self, batches: List[List[int]], shuffle: bool) -> None:
            self.batches = batches
            self.shuffle = shuffle
        def __iter__(self):
            order = list(range(len(self.batches)))
            if self.shuffle:
                import random
                random.shuffle(order)
            for bi in order:
                yield from self.batches[bi]
        def __len__(self) -> int:
            return sum(len(b) for b in self.batches)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=2, pin_memory=False)


