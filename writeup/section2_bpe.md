# CS336 Assignment 1 — Section 2: BPE Tokenizer (writeup)

## Problem (unicode1): Understanding Unicode (1 pt)

**(a)** `chr(0)` returns the NUL character `U+0000` (`'\x00'`).

**(b)** Its `__repr__()` is the escaped form `'\x00'`, whereas `print(chr(0))` emits the raw
NUL byte, which renders as nothing visible (an empty-looking line).

**(c)** Inside a string it is just an ordinary character: `"this is a test" + chr(0) + "string"`
has repr `'this is a test\x00string'` and prints as `this is a teststring` (the NUL is
invisible but still present, `len` counts it and it survives encode/decode).

## Problem (unicode2): Unicode Encodings (3 pts)

**(a)** UTF-8 gives a base vocabulary of only 256 byte values, is the dominant encoding of
web text, is ASCII-compatible (English text is 1 byte/char, so byte sequences are short),
and has no byte-order/endianness issue. UTF-16/UTF-32 pad every ASCII character with zero
bytes (`"hello".encode("utf-16")` is 12 bytes incl. BOM, UTF-32 is 24 bytes), so sequences
get 2–4× longer and are dominated by `\x00` bytes, which wastes vocabulary/model capacity.

**(b)** The function decodes each byte independently, but UTF-8 encodes non-ASCII code
points as multi-byte sequences whose individual bytes are not valid on their own.
Example: `"é".encode("utf-8") == b'\xc3\xa9'`; `decode_utf8_bytes_to_str_wrong(b'\xc3\xa9')`
raises `UnicodeDecodeError` (`bytes([0xC3]).decode("utf-8")` fails) instead of returning
`"é"`. Same for `"牛".encode("utf-8") == b'\xe7\x89\x9b'`.

**(c)** `b'\xc3\x28'` — `0xC3` announces a 2-byte sequence, but `0x28` (`(`) is not a
continuation byte (`10xxxxxx`), so the pair is invalid UTF-8. (`b'\x80\x80'` also works:
lone continuation bytes.)

## Problem (train_bpe): BPE Tokenizer Training (15 pts)

Implementation: `cs336_basics/train_bpe.py` (+ `cs336_basics/pretokenization.py`).

* Vocabulary = special tokens + 256 bytes; merges until `vocab_size`.
* Pre-tokenization: file is chunked at `<|endoftext|>` boundaries
  (`find_chunk_boundaries`), chunks are pre-tokenized in a `multiprocessing.Pool`;
  inside a chunk the text is first `re.split` on the (escaped) special tokens so no
  merge crosses a document boundary and special tokens are never counted; each
  segment is scanned with the GPT-2 regex via `finditer`.
* Merge loop is incremental: `pair_counts`, `pair_to_words` (inverted index) and a
  lazy-deletion max-heap keyed on `(count, (bytes_a, bytes_b))` so ties break toward
  the lexicographically greater pair; each merge only rewrites the words that contain
  the pair and only re-pushes pairs whose counts changed.

Tests: `pytest tests/test_train_bpe.py` — see test log in the appendix.

## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 pts)

_TODO after running `scripts/train_bpe_cli.py --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --out-dir out/ts10k --profile`._

**(a)** time / memory / longest token: …

**(b)** profile hot-spot: …

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 pts)

_TODO (`--vocab-size 32000` on `owt_train.txt`)._

## Problem (tokenizer): Implementing the tokenizer (15 pts)

Implementation: `cs336_basics/tokenizer.py` (`Tokenizer.__init__ / from_files / encode /
encode_iterable / decode`).

* `encode`: split on special tokens (longest first, capturing group so the specials are
  kept), pre-tokenize each remaining segment, and apply merges to each pre-token by
  repeatedly merging the adjacent pair with the lowest merge rank; per-pre-token cache.
* `encode_iterable`: keeps a small text buffer and only emits the prefix that can no
  longer change: it holds back the last pre-token, the last `max(len(special))-1`
  characters (possible special-token prefix) and any trailing whitespace-only
  pre-tokens (the `\s+(?!\S)` rule looks ahead), so the stream tokenization equals the
  one-shot tokenization with O(line) memory.
* `decode`: concatenate bytes, `decode("utf-8", errors="replace")` → U+FFFD for bad bytes.

## Problem (tokenizer_experiments): Experiments with tokenizers (4 pts)

_TODO via `scripts/tokenizer_experiments.py {ratio,throughput,encode}`._

**(d)** uint16 is appropriate because vocabulary sizes of 10 000 and 32 000 are both
< 2^16 = 65 536, so every id fits in 2 bytes (half of int32) with no loss.
