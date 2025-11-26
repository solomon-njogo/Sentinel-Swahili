"""
Causal Language Modeling Dataset Utilities.
Transforms raw Swahili text into fixed-length token blocks suitable for
auto-regressive fine-tuning (e.g., RoBERTa converted to a decoder).
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .tokenizer_utils import load_tokenizer

logger = logging.getLogger(__name__)


class CausalLMChunkedDataset(Dataset):
    """
    Concatenate corpora and slice into overlapping token blocks for causal LM.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_name: str = "roberta-base",
        block_size: int = 512,
        stride: int = 0,
        pad_to_block: bool = True,
        shuffle_chunks: bool = True,
    ):
        if tokenizer is None:
            tokenizer = load_tokenizer(tokenizer_name)

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.tokenizer = tokenizer
        self.block_size = min(block_size, tokenizer.model_max_length)
        self.stride = stride if stride is not None else 0
        self.pad_to_block = pad_to_block
        self.shuffle_chunks = shuffle_chunks

        logger.info(
            "Building causal dataset: %d texts, block_size=%d, stride=%d, pad=%s",
            len(texts),
            self.block_size,
            self.stride,
            pad_to_block,
        )

        self.examples = self._tokenize_and_chunk(texts)
        logger.info("Created %d causal LM blocks", len(self.examples))

    def _tokenize_and_chunk(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        eos_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else self.tokenizer.sep_token_id
        )
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else eos_id
        )
        if pad_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

        all_token_ids: List[int] = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            all_token_ids.extend(encoded["input_ids"])
            if eos_id is not None:
                all_token_ids.append(eos_id)

        examples: List[Dict[str, torch.Tensor]] = []
        step = self.block_size if self.stride <= 0 else max(1, self.block_size - self.stride)

        for start in range(0, len(all_token_ids), step):
            end = start + self.block_size
            chunk = all_token_ids[start:end]

            if len(chunk) < self.block_size:
                if not self.pad_to_block:
                    break
                pad_length = self.block_size - len(chunk)
                attention_mask = [1] * len(chunk) + [0] * pad_length
                chunk = chunk + [pad_id] * pad_length
            else:
                attention_mask = [1] * self.block_size
                pad_length = 0

            labels = chunk.copy()
            if pad_length:
                labels[-pad_length:] = [-100] * pad_length

            examples.append(
                {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

        if self.shuffle_chunks:
            random.shuffle(examples)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

