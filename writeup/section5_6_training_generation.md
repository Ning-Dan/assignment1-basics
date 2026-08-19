# Section 5 – Training loop, Section 6 – Generating text

Code: `cs336_basics/data.py`, `cs336_basics/serialization.py`, `cs336_basics/train.py`
(training CLI, run as `python -m cs336_basics.train`), `cs336_basics/decoding.py`,
`scripts/generate.py`.  Tests: `tests/test_data.py`, `tests/test_serialization.py`
(both pass: `2 passed`).

## 5.1 Data loading (`data.py`)

`get_batch(dataset, batch_size, context_length, device)` draws `batch_size` start
indices `i ~ Uniform{0, …, n − m − 1}` (`n = len(dataset)`, `m = context_length`),
gathers one `(B, m+1)` window with a single fancy-index (`dataset[starts[:, None] +
arange(m+1)]`, which copies only those rows out of a memmap), and returns
`x = window[:, :-1]`, `y = window[:, 1:]` as `int64` tensors moved to `device`.
An optional `numpy.random.Generator` argument makes sampling reproducible (used
for the fixed validation batches).  `load_token_file(path, dtype=uint16)` opens
`.npy` files with `np.load(mmap_mode="r")` and raw files with `np.memmap`, so
neither train nor validation set is ever read into RAM; `get_batch` cost is
~0.4 ms for B=16, m=64 (measured on CPU).

## 5.2 Checkpointing (`serialization.py`)

`save_checkpoint(model, optimizer, iteration, out, extra=None)` writes
`{"model": state_dict, "optimizer": state_dict, "iteration": int, "extra": …}` with
`torch.save` (path or file-like).  `load_checkpoint(src, model, optimizer)`
restores both state dicts and returns `iteration`.  Two details: a
`torch.compile`d model is unwrapped (`_orig_mod`) before `state_dict()` so the
keys carry no `_orig_mod.` prefix and load into a plain model; tensors are loaded
with `map_location="cpu"` and moved to the parameter device by `load_state_dict`,
so a GPU checkpoint loads on CPU and vice-versa.  `extra` carries the run config,
`elapsed_s` and `tokens_seen`, which is what makes `--resume` continue the
wall-clock axis of the loss curve instead of restarting it at 0.

## 5.3 Training script (`train.py`)

`python -m cs336_basics.train --train-data … --val-data … --out-dir RUN [flags]`.
Full flag list: `--help` (also reproduced in the report).  Design:

* **Model** is `cs336_basics.model.TransformerLM(vocab_size, context_length,
  d_model, num_layers, num_heads, d_ff, rope_theta)`.  §7 ablation flags map onto
  the model's knobs: `--no-rmsnorm → norm="none"`, `--post-norm →
  norm_position="post"`, `--no-rope → use_rope=False`, `--ffn silu → ffn_type="silu"`
  (pass `--d-ff 4*d_model` yourself, as the assignment asks).  Non-default knob
  values are only forwarded if the constructor accepts them, otherwise the script
  exits with a message naming the missing parameter (`--tie-embeddings` is
  reserved this way; model.py does not implement it).
* **Optimizer / schedule**: `cs336_basics.optimizer.AdamW` (`--lr --beta1 --beta2
  --eps --weight-decay`); learning rate set every step from
  `get_lr_cosine_schedule(step, lr, min_lr, warmup_iters, cosine_iters)` with
  `min_lr` defaulting to `lr/10` and `cosine_iters` defaulting to `--max-steps`, so
  decay ends exactly at the last step; global-norm gradient clipping
  (`--grad-clip`, default 1.0) via `nn_utils.gradient_clipping`.
* **Budget**: `--max-steps N` or `--total-tokens T` (steps = T / (B·m), the form
  the assignment uses: 327,680,000 tokens); `--max-wallclock-min` stops (and
  checkpoints) after a time limit — the leaderboard's 45-min cap.
* **Data**: memmapped as above; a sanity check aborts if the first 1M tokens contain
  an id ≥ `--vocab-size` (catches wrong dtype / wrong vocab).
* **Evaluation**: every `--eval-interval` steps (and at the end) the mean
  cross-entropy over `--eval-batches` validation batches drawn with a fixed seed
  (`--eval-seed`), so every evaluation and every run sees the same validation
  tokens and curves are comparable.  Eval time is excluded from the throughput
  numbers but included in `elapsed_s`.
* **Logging**: every `--log-interval` steps one JSON line is appended to
  `RUN/log.jsonl` and echoed to the console.  Fields: `step, train_loss, lr,
  elapsed_s (wall-clock incl. previous segments on resume), tokens_seen,
  tokens_per_s, step_ms, grad_norm` (pre-clip global norm), plus `val_loss,
  val_perplexity` on eval steps.  `RUN/config.json` records all arguments and the
  parameter count.  `--wandb` mirrors the same records to Weights & Biases
  (`--wandb-project`); the import happens only when the flag is given, so no
  network dependency by default.
* **Checkpoints**: `RUN/ckpt_step<N>.pt` every `--ckpt-interval` steps and
  `RUN/ckpt_final.pt` at the end (also on Ctrl-C or wall-clock stop);
  `--resume PATH` restores model, optimizer, step, elapsed time and tokens seen.
* **Speed knobs**: `--compile` (torch.compile; verified on CPU), `--dtype bf16`
  (autocast on CUDA; loss is computed in fp32 from up-cast logits), `--tf32`.
* **Debugging**: `--overfit-one-batch` reuses the first batch; with the 1.4M-param
  smoke model the loss goes 9.2 → 0.006 in 200 steps, the standard sanity check
  that the model/loss/optimizer path is wired correctly.

Smoke test (CPU, d_model 64, 2 layers, ctx 64, B 16, TinyStories-valid subset
tokenized with the 10k vocab): loss 9.03 → 7.03 over 60 steps; resuming from
`ckpt_step40.pt` continued at step 40 with elapsed 96 s and reached step 70;
`generate.py` produced text from the resulting checkpoint.  Practical note for
the shared C12 box: with 4 OMP threads on cores already busy with other jobs each
tiny op cost ~9 ms of thread synchronisation (fwd 667 ms vs 70 ms single-threaded),
so use `OMP_NUM_THREADS=1` for CPU smoke tests there.

## 6 Decoding (`decoding.py`, `scripts/generate.py`)

`generate(model, prompt_ids, max_new_tokens, temperature, top_p, eos_token_id,
device, context_length, generator)` loops: feed the last `context_length` tokens,
take the last row of logits `v`, sample, append; stop at `eos_token_id`
(`<|endoftext|>`) or after `max_new_tokens`; returns the new ids only.

* Temperature (eq. 23): `softmax(v/τ)` computed as `exp((v/τ) − max) / Σ`;
  `temperature == 0` takes the argmax (the τ→0 limit).
* Top-p (eq. 24): sort `q` descending, cumulative sum `c`; drop token `j` iff
  `c_j − q_j ≥ p` (the mass *before* it already reaches `p`), which keeps the
  smallest prefix whose mass is ≥ p including the token that crosses the
  threshold; the top-1 token is always kept; scatter back and renormalise.
  Checked: `q=(.5,.3,.15,.05)`: p=0.5 → (1,0,0,0); p=0.8 → (.625,.375,0,0);
  p=0.81 → (.526,.316,.158,0); sampling 1000 draws with p=0.8 only ever returned
  ids {0,1}.
* Sampling is `torch.multinomial` with an optional `torch.Generator` for
  reproducibility.  `generate_text(model, tokenizer, prompt, …)` is the string
  wrapper.

`scripts/generate.py --checkpoint CKPT --vocab vocab.json --merges merges.txt
--prompt "…" --max-new-tokens 256 --temperature T --top-p P [--seed S]
[--num-samples K] [--device]` rebuilds the model from the config stored in the
checkpoint (`extra.model_config`; overridable per flag), loads weights, and prints
prompt + continuation, with token count / stop reason on stderr.

## §7 `generate` deliverable

*(To be filled by the §7 experiments: ≥256-token dump from the trained TinyStories
checkpoint via the command above, plus commentary on fluency and on at least two
factors — e.g. temperature/top-p, model size / training tokens, validation loss.)*
