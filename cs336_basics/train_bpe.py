"""Byte-level BPE tokenizer training.

Algorithm
---------
1. Vocabulary init: special tokens + all 256 byte values.
2. Pre-tokenize the corpus (GPT-2 regex, special tokens as hard boundaries,
   parallel over file chunks) and count each distinct pre-token.
3. Repeatedly merge the most frequent adjacent pair of tokens (ties broken by the
   lexicographically greater ``(bytes, bytes)`` pair) until ``vocab_size`` is reached
   or no pairs remain.

The merge loop is incremental: we keep

* ``pair_counts``   : (a, b) -> total frequency
* ``pair_to_words`` : (a, b) -> set of word indices containing the pair
* a max-heap over pairs (lazy deletion) so picking the best pair is O(log P)

so each merge only touches the words that actually contain the merged pair.
"""

from __future__ import annotations

import heapq
import os
from collections import defaultdict

from cs336_basics.pretokenization import count_pretokens_in_file


class _HeapItem:
    """Heap entry ordering: higher count first, then lexicographically greater pair first.

    ``heapq`` is a min-heap, so ``__lt__`` is inverted accordingly.
    """

    __slots__ = ("count", "key", "pair")

    def __init__(self, count: int, key: tuple[bytes, bytes], pair: tuple[int, int]):
        self.count = count
        self.key = key
        self.pair = pair

    def __lt__(self, other: _HeapItem) -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.key > other.key


def _adjacent_pairs(word: list[int]) -> list[tuple[int, int]]:
    return list(zip(word[:-1], word[1:]))


def _merge_word(word: list[int], a: int, b: int, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of (a, b) in ``word`` with ``new_id``."""
    out: list[int] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Args:
        input_path: path to a UTF-8 text file.
        vocab_size: final vocabulary size (bytes + merges + special tokens).
        special_tokens: strings that are added to the vocabulary verbatim and act as
            hard boundaries during training (never merged, never counted).
        num_processes: worker processes for pre-tokenization (default: all CPUs
            available to this process).

    Returns:
        (vocab, merges): ``vocab`` maps token id -> bytes; ``merges`` is the list of
        merged pairs in creation order.
    """
    special_tokens = list(dict.fromkeys(special_tokens or []))  # dedupe, keep order

    # ---- 1. vocabulary initialisation --------------------------------------
    vocab: dict[int, bytes] = {}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])
    byte_offset = len(special_tokens)  # token id of byte value b is byte_offset + b

    num_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []
    if num_merges <= 0:
        return vocab, merges

    # ---- 2. pre-tokenisation ------------------------------------------------
    pretoken_counts = count_pretokens_in_file(input_path, special_tokens, num_processes=num_processes)

    words: list[list[int]] = []
    counts: list[int] = []
    for tok, cnt in pretoken_counts.items():
        words.append([byte_offset + b for b in tok])
        counts.append(cnt)
    del pretoken_counts

    # ---- 3. build pair statistics ------------------------------------------
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for idx, word in enumerate(words):
        cnt = counts[idx]
        for pair in _adjacent_pairs(word):
            pair_counts[pair] += cnt
            pair_to_words[pair].add(idx)

    heap: list[_HeapItem] = [
        _HeapItem(cnt, (vocab[a], vocab[b]), (a, b)) for (a, b), cnt in pair_counts.items() if cnt > 0
    ]
    heapq.heapify(heap)

    def push(pair: tuple[int, int]) -> None:
        cnt = pair_counts.get(pair, 0)
        if cnt > 0:
            heapq.heappush(heap, _HeapItem(cnt, (vocab[pair[0]], vocab[pair[1]]), pair))

    # ---- 4. merge loop ------------------------------------------------------
    while len(merges) < num_merges and heap:
        # pop until we find an entry whose count is still current (lazy deletion)
        item = heapq.heappop(heap)
        current = pair_counts.get(item.pair, 0)
        if current <= 0 or current != item.count:
            continue

        a, b = item.pair
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        touched: set[tuple[int, int]] = set()
        for idx in list(pair_to_words.get((a, b), ())):
            word = words[idx]
            cnt = counts[idx]
            old_pairs = _adjacent_pairs(word)
            if (a, b) not in old_pairs:  # stale membership
                pair_to_words[(a, b)].discard(idx)
                continue
            new_word = _merge_word(word, a, b, new_id)
            new_pairs = _adjacent_pairs(new_word)

            for p in old_pairs:
                pair_counts[p] -= cnt
                touched.add(p)
            for p in new_pairs:
                pair_counts[p] += cnt
                pair_to_words[p].add(idx)
                touched.add(p)
            new_pair_set = set(new_pairs)
            for p in set(old_pairs):
                if p not in new_pair_set:
                    pair_to_words[p].discard(idx)
            words[idx] = new_word

        # the merged pair is gone for good
        pair_counts.pop((a, b), None)
        pair_to_words.pop((a, b), None)
        touched.discard((a, b))
        for p in touched:
            if pair_counts.get(p, 0) <= 0:
                pair_counts.pop(p, None)
                pair_to_words.pop(p, None)
            else:
                push(p)

    return vocab, merges


__all__ = ["train_bpe"]
