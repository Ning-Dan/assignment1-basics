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

Run: `scripts/train_bpe_cli.py --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000
--out-dir out/ts10k --profile` (2.23 GB, 2 717 699 documents) on a shared 20-core box,
pinned to **4 cores** (`taskset -c 0-3`, `nice 15`), Python 3.13.

**(a)** `train_bpe` took **217 s** (3 min 37 s wall including start-up and serialization);
peak memory **716 MB** (`/usr/bin/time -v` max RSS: 109 MB in the main process, 700 MB in
the largest pre-tokenization worker). The longest token is the 15-byte
`b' accomplishment'`; the next longest are `' disappointment'`, `' responsibility'`,
`' uncomfortable'`, `' compassionate'`, `' understanding'`, `' neighbourhood'` — all
ordinary long English words with a leading space, i.e. exactly what a whole-word BPE on
children's stories should learn (no junk, no punctuation runs). Yes, this makes sense.

**(b)** cProfile: `count_pretokens_in_file` (pre-tokenization) = **213 s of 217 s (98 %)**;
the incremental merge loop for 9 743 merges is only ~4 s because TinyStories has few
distinct pre-tokens. Pre-tokenization is a Python-level loop over ~500 M regex matches, so
it is CPU-bound and scales with the number of worker processes (we were capped at 4;
with all 20 cores it would be well under a minute). After this run we replaced the
per-match `finditer` loop by `Counter.update(pattern.findall(doc))` (both run in C) —
the 5 M-byte sample went from 0.9 s to 0.3 s and the outputs are byte-identical.

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 pts)

Run: `scripts/train_bpe_cli.py --input data/owt_train.txt --vocab-size 32000 --out-dir
out/owt32k --profile` on the full 11.9 GB `owt_train.txt` (2 355 962 documents), same 4-core
pin. (A 1 GB-prefix dry run took 188 s / 2.1 GB peak, which is how we decided the full run
was affordable.)

**(a)** `train_bpe` took **1 523 s (25.4 min)**; peak memory **9.5 GB** (max RSS of the main
process — the merge-loop state: every distinct pre-token as a `list[int]`, the
pair → count table, the pair → word inverted index and the lazy heap). Profile: pre-tokenization 460 s
(30 %), merge loop ~1 050 s (70 %) — the opposite split from TinyStories, because OWT has
far more distinct pre-tokens (the main process holds 9.5 GB of state vs 0.1 GB) and 3× more merges, and each early merge
touches millions of words. The longest token (64 bytes) is
`b'\xc3\x83\xc3\x82' * 16` = `ÃÂÃÂÃÂ…`, i.e. mojibake — the residue of text that was UTF-8-encoded more than
once — which appears verbatim as long runs in many scraped pages; the next longest are
`'-' * 64`, `'—' * 16`, `'_' * 32`, `'=' * 32`, `'.' * 32`, `'*' * 32` (ASCII-art
separators). This is *not* a linguistically meaningful token, but it does make sense for
byte-level BPE on web text: these long identical byte runs occur thousands of times and
every merge inside them halves their length, so they win the frequency race all the way to
64 bytes. The longest *alphabetic* tokens are `' disproportionately'`,
`' telecommunications'`, `' environmentalists'`, `' unconstitutional'`,
`' cryptocurrencies'`, `' counterterrorism'`.

**(b)** TinyStories (10 K) vs OpenWebText (32 K) vocabularies:

* Overlap: 7 319 tokens are in both — 73 % of the TinyStories vocab, 23 % of the OWT vocab.
  Almost every TS token that OWT lacks is a children's-story noun/verb
  (`' granddaughter'`, `' marshmallows'`, `' caterpillars'`, `' veterinarian'`,
  `' storekeeper'`, `' superheroes'`, `' blueberries'`).
* Both are ~95 % whole words with a leading space (TS 97 %, OWT 95 %); OWT has ~10× more
  digit-containing tokens (582 vs 16 — years, prices, dates) and 4× more non-ASCII tokens
  (559 vs 148: curly quotes, `—`, `…`, mojibake).
* OWT contains web/news-specific tokens that TS never sees: `' https'`, `' http'`,
  `' www'`, `'html'`, `' href'`, `' Advertisement'`, `' Copyright'`, `' Comments'`,
  `' Share'`, `' Click'`, `' Getty'`, `' Reuters'`, `' Twitter'`, `' Facebook'`,
  `' Google'`, `'realDonaldTrump'`, `' Obamacare'`, `'################'`.
* TS long tokens are all emotional/descriptive vocabulary of stories
  (`' accomplishment'`, `' disappointment'`, `' compassionate'`); OWT long tokens are
  political/technical (`' unconstitutional'`, `' telecommunications'`) plus the
  separator/mojibake runs above.

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

All numbers from `scripts/tokenizer_experiments.py` (`ratio` samples 10 documents with
`random.Random(seed)` from the first 200 MB of the validation file; `throughput` encodes
a whole file with `Tokenizer.encode`; `encode` writes a uint16 `.npy`).

**(a)** Compression ratio (bytes / token) on 10 sampled documents:

| tokenizer | data | seed 0 | seed 1 |
|---|---|---|---|
| TinyStories 10 K | TinyStories valid | **4.10** (7 633 B → 1 861 tok) | 4.05 |
| OpenWebText 32 K | OpenWebText valid | **4.52** (30 001 B → 6 642 tok) | 4.43 |

Over the *whole* validation sets (from (d)): TS 4.12 B/token, OWT 4.37 B/token. So each
tokenizer gets ~4.1–4.5 bytes per token on its own domain; the 32 K vocabulary buys a
slightly better ratio on the much more diverse web text.

**(b)** TinyStories tokenizer on OpenWebText documents: **3.27 B/token** (seed 0; 3.43 for
seed 1) vs 4.52 with the OWT tokenizer — 30–40 % more tokens for the same bytes. Qualitatively the text gets
chopped into sub-word pieces: e.g. `' New Jersey judge … official misconduct, issuing a
criminal summons'` → TS: `' Ne' 'w' ' J' 'er' 'sey'`, `' offic' 'ial'`, `' mis' 'cond'
'uct'`, `' is' 's' 'uing'`, `' cr' 'im' 'inal'`, `' s' 'umm' 'ons'` (87 tokens for 300 B),
while the OWT tokenizer keeps `' Jersey'`, `' official'`, `' misconduct'`, `' issuing'`,
`' criminal'` whole (60 tokens). On the 10 OWT docs 21 % of the TS tokenizer's output
tokens are single bytes and 40 % are ≤ 2 bytes, vs 15 % / 23 % for the OWT tokenizer.
(The reverse direction is much milder: OWT tokenizer on TS docs = 3.96 B/token, since the
larger web vocabulary already covers most children's-story words.)

**(c)** Throughput of `Tokenizer.encode` (single process, one pinned core, Python 3.13):

| tokenizer | text | throughput |
|---|---|---|
| TS 10 K | `tinystories_sample_5M.txt` (5.2 MB) | 5.4 MB/s |
| TS 10 K | TS valid (22.5 MB) | 7.3 MB/s |
| OWT 32 K | OWT valid, first 50 MB | 3.3 MB/s |

TinyStories is highly repetitive so the per-pre-token cache hits most of the time; on
web text the tokenizer runs at ~3.3 MB/s. At 3.3–7 MB/s the 825 GB Pile would take
**~33–70 h single-process** (≈ 2–4 h on 20 cores with the same document-level chunking we
use for `encode --num-processes`).

**(d)** Encoded datasets (uint16 `.npy`, verified with `np.load(mmap_mode="r")`:
`max(id) < vocab_size`, `<|endoftext|>` count equals the count in the text file):

| file | tokens | bytes/token | wall time |
|---|---|---|---|
| `ts_valid_10k.npy` | 5 465 883 | 4.12 | 5 s (`encode_iterable`, 64 MB RSS) |
| `ts_train_10k.npy` | 541 229 347 | 4.12 | 700 s (`encode_iterable`, 1 core, 196 MB RSS) |
| `owt_valid_32k.npy` | 66 401 098 | 4.37 | 57 s (4 processes) |
| `owt_train_32k.npy` | 2 727 120 452 | 4.37 | 1 848 s (4 processes) |

The 2.2 GB TinyStories train set was encoded with the streaming `encode_iterable` path in
constant memory (~200 MB RSS); the 12 GB OWT train set with 4 worker processes that each
encode a chunk delimited at `<|endoftext|>` boundaries (identical output to one-shot
`encode`, checked on the 5 M sample). uint16 is appropriate because vocabulary sizes of
10 000 and 32 000 are both < 2^16 = 65 536, so every id fits in 2 bytes (half of int32)
with no loss; the token arrays are 1.0 GB (TS) and 5.5 GB (OWT) instead of 2×
that as int32.

## Appendix: commands & raw logs

Environment: C12 shared box, Python 3.13.15 (uv-managed), torch 2.11.0+cu130, `regex`,
`tiktoken`; every job pinned with `nice -n 15 taskset -c 0-3`, `OMP_NUM_THREADS=4`.
Full logs are in `/home/c12/workspace/claude_workspace/cs336_a1/logs/`.

```
$ .venv/bin/python -m pytest tests/test_train_bpe.py tests/test_tokenizer.py -q
======================== 27 passed, 1 xpassed in 3.57s =========================
# (test_encode_memory_usage is marked xfail; it happens to pass here)

$ /usr/bin/time -v .venv/bin/python scripts/train_bpe_cli.py --input data/TinyStoriesV2-GPT4-train.txt \
      --vocab-size 10000 --out-dir out/ts10k --profile
vocab_size=10000 merges=9743
time=217.2s peak_rss(main)=109MB peak_rss(max child)=700MB
longest token (15 bytes): b' accomplishment'
        1    0.000    0.000  213.310  213.310 cs336_basics/pretokenization.py:118(count_pretokens_in_file)
	Elapsed (wall clock) time (h:mm:ss or m:ss): 3:37.23
	Maximum resident set size (kbytes): 716352

$ /usr/bin/time -v .venv/bin/python scripts/train_bpe_cli.py --input owt_train_1G.txt --vocab-size 32000 ...   # 1 GB dry run
time=188.1s peak_rss(main)=2059MB peak_rss(max child)=436MB
	Maximum resident set size (kbytes): 2108032

$ ulimit -v 22000000; /usr/bin/time -v .venv/bin/python scripts/train_bpe_cli.py --input data/owt_train.txt \
      --vocab-size 32000 --out-dir out/owt32k --profile
vocab_size=32000 merges=31743
time=1523.0s peak_rss(main)=9302MB peak_rss(max child)=473MB
longest token (64 bytes): b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82...' (x16)
        1  428.950  428.950 1513.812 1513.812 cs336_basics/train_bpe.py:68(train_bpe)
        1    0.000    0.000  459.649  459.649 cs336_basics/pretokenization.py:122(count_pretokens_in_file)
	Elapsed (wall clock) time (h:mm:ss or m:ss): 25:23.44
	Maximum resident set size (kbytes): 9525604

$ scripts/tokenizer_experiments.py ratio --vocab out/ts10k/vocab.json --merges out/ts10k/merges.txt --input data/TinyStoriesV2-GPT4-valid.txt
10 docs, 7633 bytes, 1861 tokens -> 4.102 bytes/token
$ ... ratio (TS10k) --input data/owt_valid.txt
10 docs, 30001 bytes, 9167 tokens -> 3.273 bytes/token
$ ... ratio (OWT32k) --input data/owt_valid.txt
10 docs, 30001 bytes, 6642 tokens -> 4.517 bytes/token
$ ... ratio (OWT32k) --input data/TinyStoriesV2-GPT4-valid.txt
10 docs, 7633 bytes, 1927 tokens -> 3.961 bytes/token

$ ... throughput (TS10k) --input tests/fixtures/tinystories_sample_5M.txt
5242880 bytes -> 1274107 tokens in 0.98s = 5.36 MB/s ; Pile (825GB) would take ~42.7 h single-process
$ ... throughput (TS10k) --input data/TinyStoriesV2-GPT4-valid.txt
22502601 bytes -> 5465883 tokens in 3.06s = 7.34 MB/s ; Pile ~31.2 h
$ ... throughput (OWT32k) --input owt_valid_50M.txt
50000000 bytes -> 11458749 tokens in 15.30s = 3.27 MB/s ; Pile ~70.1 h

$ ... encode (TS10k) --input data/TinyStoriesV2-GPT4-valid.txt --output data/ts_valid_10k.npy
wrote 5465883 tokens (uint16) to data/ts_valid_10k.npy in 5s          (max RSS 63652 KB)
$ ... encode (TS10k) --input data/TinyStoriesV2-GPT4-train.txt --output data/ts_train_10k.npy
wrote 541229347 tokens (uint16) to data/ts_train_10k.npy in 700s      (max RSS 195932 KB)
$ ... encode (OWT32k) --input data/owt_valid.txt --output data/owt_valid_32k.npy --num-processes 4 --num-chunks 128
wrote 66401098 tokens (uint16) to data/owt_valid_32k.npy in 56s       (max RSS 183900 KB)
$ ... encode (OWT32k) --input data/owt_train.txt --output data/owt_train_32k.npy --num-processes 4 --num-chunks 1024
wrote 2727120452 tokens (uint16) to data/owt_train_32k.npy in 1848s (max RSS 206748 KB, wall 30:49)

$ python -c "np.load(f, mmap_mode='r')"   # sanity
data/ts_valid_10k.npy  uint16 (5465883,)   max 9999  < 10000  eot_count 27630
data/ts_train_10k.npy  uint16 (541229347,) max 9999  < 10000  eot_count 2717699
data/owt_valid_32k.npy uint16 (66401098,)  max 31999 < 32000  eot_count 59059
data/owt_train_32k.npy uint16 (2727120452,) max 31999 < 32000 eot_count 2399397
```
