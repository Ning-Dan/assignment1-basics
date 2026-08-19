"""Byte-level BPE tokenizer: encode text -> token ids, decode ids -> text.

Encoding mirrors training: split on special tokens, pre-tokenize each segment
with the GPT-2 regex, then apply the learned merges (in creation order) inside
each pre-token.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from functools import lru_cache

import regex as re

from cs336_basics.pretokenization import PAT, special_token_splitter

_PAT_RE = re.compile(PAT)


# --------------------------------------------------------------------------- #
# GPT-2 style byte <-> printable-unicode mapping used for on-disk serialization
# --------------------------------------------------------------------------- #
@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _bytes_to_printable(b: bytes) -> str:
    table = gpt2_bytes_to_unicode()
    return "".join(table[x] for x in b)


def _printable_to_bytes(s: str) -> bytes:
    inv = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    return bytes(inv[ch] for ch in s)


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_filepath: str | os.PathLike,
    merges_filepath: str | os.PathLike,
) -> None:
    """Serialize in the GPT-2 text format (vocab.json: token-string -> id; merges.txt: 'a b' per line)."""
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump({_bytes_to_printable(tok): idx for idx, tok in vocab.items()}, f, ensure_ascii=False, indent=1)
    with open(merges_filepath, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_bytes_to_printable(a)} {_bytes_to_printable(b)}\n")


def load_vocab_and_merges(
    vocab_filepath: str | os.PathLike,
    merges_filepath: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_filepath, encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {int(idx): _printable_to_bytes(tok) for tok, idx in raw.items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#version"):
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                continue
            merges.append((_printable_to_bytes(parts[0]), _printable_to_bytes(parts[1])))
    return vocab, merges


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
class Tokenizer:
    """BPE tokenizer built from a vocabulary + ordered merge list (+ optional special tokens)."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.special_tokens: list[str] = list(dict.fromkeys(special_tokens or []))

        # make sure every special token has an id
        existing = {tok: idx for idx, tok in self.vocab.items()}
        for st in self.special_tokens:
            b = st.encode("utf-8")
            if b not in existing:
                new_id = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[new_id] = b
                existing[b] = new_id

        self.bytes_to_id: dict[bytes, int] = {tok: idx for idx, tok in self.vocab.items()}
        self.special_to_id: dict[str, int] = {st: self.bytes_to_id[st.encode("utf-8")] for st in self.special_tokens}
        # rank of each merge = its position in the merge list (lower = applied first)
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}

        self._special_splitter = special_token_splitter(self.special_tokens, capture=True)
        self._max_special_len = max((len(s) for s in self.special_tokens), default=0)
        self._cache: dict[bytes, list[int]] = {}
        self._cache_limit = 1 << 15

    # ------------------------------------------------------------------ #
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab, merges = load_vocab_and_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    # ------------------------------------------------------------------ #
    def _bpe(self, pretoken: bytes) -> list[int]:
        """Apply merges to one pre-token and return its token ids."""
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached

        parts: list[bytes] = [bytes([b]) for b in pretoken]
        ranks = self.merge_ranks
        while len(parts) > 1:
            # find the adjacent pair with the lowest merge rank
            best_rank = None
            best_pair = None
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = (parts[i], parts[i + 1])
            if best_pair is None:
                break
            a, b = best_pair
            merged = a + b
            new_parts: list[bytes] = []
            i = 0
            n = len(parts)
            while i < n:
                if i < n - 1 and parts[i] == a and parts[i + 1] == b:
                    new_parts.append(merged)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        ids = [self.bytes_to_id[p] for p in parts]
        if len(self._cache) < self._cache_limit:
            self._cache[pretoken] = ids
        return ids

    def _encode_segment(self, text: str) -> Iterator[int]:
        """Encode text that contains no special tokens."""
        for m in _PAT_RE.finditer(text):
            yield from self._bpe(m.group().encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self._special_splitter is None:
            ids.extend(self._encode_segment(text))
            return ids
        for segment in self._special_splitter.split(text):
            if not segment:
                continue
            sid = self.special_to_id.get(segment)
            if sid is not None:
                ids.append(sid)
            else:
                ids.extend(self._encode_segment(segment))
        return ids

    # ------------------------------------------------------------------ #
    def _safe_cut(self, buffer: str) -> int:
        """Largest prefix length of ``buffer`` that can be encoded now without
        risking a different tokenization once more text arrives.

        We tile the buffer into units (special tokens / pre-tokens).  Rules:
        * hold back the last unit (it may grow when more text arrives);
        * hold back the last ``max_special_len - 1`` chars (possible special-token prefix);
        * never cut right after a whitespace-only unit: ``\\s+(?!\\S)`` decides how much
          whitespace to take by looking at what follows, so trailing whitespace of a
          prefix tokenizes differently than the same whitespace inside the full text.
        """
        limit = len(buffer) - max(self._max_special_len - 1, 0)
        if limit <= 0:
            return 0
        segments = self._special_splitter.split(buffer) if self._special_splitter is not None else [buffer]
        starts: list[int] = []
        is_ws: list[bool] = []
        pos = 0
        for segment in segments:
            if not segment:
                continue
            if segment in self.special_to_id:
                starts.append(pos)
                is_ws.append(False)
            else:
                for m in _PAT_RE.finditer(segment):
                    starts.append(pos + m.start())
                    is_ws.append(m.group().isspace())
            pos += len(segment)
        if len(starts) < 2:
            return 0
        # candidate: last unit whose start is within the limit, but never the very last unit
        i = len(starts) - 1
        while i > 0 and starts[i] > limit:
            i -= 1
        # do not cut right after whitespace-only units
        while i > 0 and is_ws[i - 1]:
            i -= 1
        return starts[i] if i > 0 else 0

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode a stream of strings (e.g. lines of a file) with constant memory."""
        buffer = ""
        for chunk in iterable:
            if not chunk:
                continue
            buffer += chunk
            cut = self._safe_cut(buffer)
            if cut > 0:
                yield from self.encode(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer)

    # ------------------------------------------------------------------ #
    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


__all__ = ["Tokenizer", "gpt2_bytes_to_unicode", "load_vocab_and_merges", "save_vocab_and_merges"]
