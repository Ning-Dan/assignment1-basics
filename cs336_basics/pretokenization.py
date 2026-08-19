"""Pre-tokenization utilities shared by BPE training and the BPE tokenizer.

Pre-tokenization = split raw text into coarse "words" (pre-tokens) with the GPT-2
regex.  BPE merges never cross pre-token boundaries, and special tokens act as hard
boundaries that are never merged with anything.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

# GPT-2 pre-tokenization pattern (the "slightly prettier" tiktoken form).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(PAT)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Chunk the file into parts that can be pre-tokenized independently.

    Every boundary (except 0 and EOF) is placed at the *start* of an occurrence of
    ``split_special_token``, so no pre-token / merge can straddle a chunk boundary.
    May return fewer chunks than requested if boundaries collide.
    (Adapted from the course-provided ``pretokenization_example.py``.)
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":  # EOF
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def special_token_splitter(special_tokens: list[str] | None, capture: bool = False) -> re.Pattern | None:
    """Regex that splits text on any special token.

    Longer tokens are tried first so that overlapping specials
    (e.g. ``<|eot|><|eot|>`` vs ``<|eot|>``) resolve to the longest match.
    With ``capture=True`` the special tokens themselves are kept in the split output.
    """
    if not special_tokens:
        return None
    alternation = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    return re.compile(f"({alternation})" if capture else alternation)


def iter_pretokens(text: str) -> Iterator[str]:
    """Yield GPT-2 pre-tokens of ``text`` (no special-token handling)."""
    for m in _PAT_RE.finditer(text):
        yield m.group()


def count_pretokens_in_text(text: str, special_tokens: list[str] | None) -> Counter[bytes]:
    """Count pre-tokens (as UTF-8 byte strings) in ``text``.

    Special tokens are used as hard segment boundaries and are *not* counted.
    """
    counts: Counter[bytes] = Counter()
    splitter = special_token_splitter(special_tokens)
    segments = splitter.split(text) if splitter is not None else [text]
    for segment in segments:
        if not segment:
            continue
        for m in _PAT_RE.finditer(segment):
            counts[m.group().encode("utf-8")] += 1
    return counts


def _count_chunk(args: tuple[str | os.PathLike, int, int, list[str] | None]) -> Counter[bytes]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return count_pretokens_in_text(chunk, special_tokens)


def default_num_processes() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        n = os.cpu_count() or 1
    return max(1, n)


def count_pretokens_in_file(
    input_path: str | os.PathLike,
    special_tokens: list[str] | None,
    num_processes: int | None = None,
) -> Counter[bytes]:
    """Pre-tokenize a whole file in parallel and return pre-token -> count.

    The file is chunked at occurrences of the first special token (document
    delimiter), so chunks are independent.  If there are no special tokens the
    file is processed as a single chunk.
    """
    if num_processes is None:
        num_processes = default_num_processes()

    with open(input_path, "rb") as f:
        if special_tokens:
            boundaries = find_chunk_boundaries(f, num_processes * 4, special_tokens[0].encode("utf-8"))
        else:
            f.seek(0, os.SEEK_END)
            boundaries = [0, f.tell()]

    jobs = [(input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:]) if e > s]
    total: Counter[bytes] = Counter()
    if len(jobs) <= 1 or num_processes <= 1:
        for job in jobs:
            total.update(_count_chunk(job))
        return total

    with Pool(processes=min(num_processes, len(jobs))) as pool:
        for partial in pool.imap_unordered(_count_chunk, jobs):
            total.update(partial)
    return total


__all__ = [
    "PAT",
    "count_pretokens_in_file",
    "count_pretokens_in_text",
    "find_chunk_boundaries",
    "iter_pretokens",
    "special_token_splitter",
]

