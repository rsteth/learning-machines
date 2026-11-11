# 01 / What is a token?

A small, fixed-fixture walk from text to token IDs, a card of notes, next-token scores, and text again. This is a teaching illustration of a decoder-only language model, **not a model or benchmark**.

![Five stages from text pieces to another piece](preview.svg)

Companion: [What is a token?](https://stethem.com/learning-machines/tokens/).

## Run

Node.js 18+, no dependencies, network requests or model weights:

```sh
node experiments/tokens/run.mjs
node experiments/tokens/run.mjs --subject king            # cat (default) or king
node experiments/tokens/run.mjs --verb slept              # sat (default) or slept
node experiments/tokens/run.mjs --trace                   # what the last piece heard, worked out and scored
node experiments/tokens/run.mjs --sample --seed 600       # any uint32 seed; prints each draw
node experiments/tokens/run.mjs --seed 600 --draws 500    # 500 chained draws, counted
node --test experiments/tokens/*.test.mjs
```

`--seed` and `--draws` reproduce the companion page's sampler exactly: the same seed gives the same draw (`selection.seed / 2^32`), and `--draws` chains each draw's returned seed into the next.

## What is real?

`fixture.mjs` contains actual **tiktoken 0.14.0 / cl100k_base** tokenization, including spaces. It is not Qwen tokenization. For example, `The cat sat on the` has IDs `[791, 8415, 7731, 389, 279]`; `unhappiness` splits into `un`, `h`, `appiness`. IDs are vocabulary labels, not quantities or meanings.

No tokenizer runs in the browser. `decode` only supports the individually UTF-8-decodable tokens recorded in this fixture; it rejects unknown IDs. `idFor(piece)` returns `null` for a model piece the fixture does not yet record: **the mechanics never invent an ID**. A test reports those pieces as skipped until `regenerate_fixture.py` has been run (see below).

Everything after the IDs is **hand-authored teaching data**: the notes, their directions, each piece's starting notes and match score, the work-out units, the candidates' wants and the stand-in score for the rest of the vocabulary. None of it is a trained network, embeddings from a model, measured confidence or observed output. What is real is the *shape* of the calculation: every number is computed from the step before it, and the next-token probabilities are a softmax of scores read from the final row.

## The teaching model

One row of **six stored numbers** per piece. The page calls a row a card.

**Notes are directions.** Eight named notes (`NOTES`) live in those six numbers. The six primary notes are orthogonal unit directions (the rows of the order-6 conference matrix, divided by √5), so each note is spread across five of the six numbers and no stored number stands for one idea. Two blended notes point between their ingredients (`cosy` = 0.7 cat + 0.5 sitting + 0.5 sleeping + 0.1 surface; `grand` = 0.8 royalty + 0.6 sitting). Eight notes therefore share six numbers and overlap: writing `a cat` makes `low, cosy spot` read 0.7 although nothing wrote it.

- `writeNotes(amounts)`: push the row along each note's direction (addition).
- `readNotes(row)`: how far the row lies along each direction (a dot product per note).

**1. Start.** A piece's ID looks up its starting row: `embeddingFor(piece) = writeNotes(PIECES[piece].starts)`.

**2. Listen.** For the row being followed, every earlier piece has a match score. Shares are a softmax of those scores; `amount = LISTEN_GAIN × share`; the listener adds `amount ×` the source's starting notes to its own row. A covered piece keeps its share but offers nothing, so covering one word does not make the others louder. Later pieces are never heard.

**3. Work out.** Each unit in `UNITS` reads a weighted sum of notes from the row, subtracts a threshold, keeps what is positive and writes that much of one note. `cosy` fires only when a cat is heard together with sitting or sleeping; `grand` only for royalty together with sitting. This is the one place where words combine instead of adding up.

**4. Score.** Each candidate has a score direction (`scoreDirection`, its `wants` written as six numbers). `score = row · direction`. Because the final row is the sum of its legs, each score splits exactly into one part per leg (`scoresFor(...).candidates[i].parts`). `OTHER_SCORE` is a fixed stand-in for the rest of a real vocabulary. Probabilities are `softmax(scores)`.

**View.** `viewOf(row)` is a fixed linear three-number view of the six stored numbers, used to draw the row's path. It is a projection for drawing, like plotting principal components: distances in it are not scores.

Arithmetic is rounded to ten decimal places to suppress binary floating-point tails. Tests compare equations within `1e-9`.

## What is simplified

- **Labels are ours.** A real model's directions are learned, unlabelled and far more numerous than its stored numbers. Here eight are named and six numbers hold them.
- **One round.** A real model repeats listen and work-out through many layers, and earlier pieces update their own rows before later ones read them. Here only one round runs and listeners read starting rows.
- **Match scores are authored per source piece.** In a model they come from comparing a query from the listener with a key from each source. The softmax is real; the scores are not learned. The listener's own position is left out of the softmax: its own notes are already on its row.
- **Listening is one move.** It is reported as one leg per piece, nearest piece first, so each word's contribution can be shown. The legs add up to the single move; their order is a presentation choice.
- **Two work-out units** stand in for a learned feed-forward layer. Normalisation and positional encoding are omitted, so identical pieces start identical wherever they sit.
- **The second generated token is scripted** (`.` at 100%, marked `scripted: true`), not computed.

## Mechanics API

- `createState({mode: 'greedy' | 'sample', seed: uint32, subject: 'cat' | 'king', verb: 'sat' | 'slept', covered: number[]})`: start at `tokens` with the prompt `The <subject> <verb> on the`. `covered` lists earlier piece indices to hide from the listener.
- `PHASES`: `tokens → embedding → layers → distribution → append`.
- `step(state)`: immutable stage transition. Entering `append` selects and appends one piece. Leaving it starts another cycle, or stops after two appended tokens.
- `traceFor(state, rowIndex = last)`: the followed row's `legs` (`start`, one `heard` leg per earlier piece, one `worked` leg per unit that fired), each with what it `wrote`, its six-number `delta`, the row `after` it and `viewFrom`/`viewTo`; plus `units`, `afterListening`, `output` and note readings at each stage.
- `scoresFor(state)`: the trace, every candidate's score, its direction and its per-leg parts, and `other`.
- `distributionFor(state)`: first-step probabilities computed from the scores, ending with the aggregate; the scripted second step; `[]` once stopped.
- `chooseToken(distribution, {mode, seed})`: highest-probability entry, or a deterministic LCG sample (not cryptographic randomness).
- `reset(state)`: replay with the original mode, seed and prompt.
- `decode(ids)`, `idFor(piece)`, `embeddingFor(piece)`, `readNotes`, `writeNotes`, `scoreDirection`, `viewOf`.

"All other tokens" is an aggregate, **not a token**. Choosing it stops with `outside-fixture`, leaves the prompt unchanged and appends no fake ID. With every content word covered it is the most likely entry, so even greedy selection stops there. The ordinary two-token stop is `demo-limit`, not a fabricated end-of-sequence prediction.

The website owns rendering, controls and editorial copy. It should pin both `mechanics.mjs` and `fixture.mjs` to the same commit and verify SHA-256 hashes. It must keep the hand-authored labels visible and explain `outside-fixture` without pretending an answer was generated.

## Reproduce the tokenizer fixture

This optional verification installs a tokenizer, **not a model**. Its first encoding load may download a tokenizer vocabulary. Normal demos and tests do not download anything.

```sh
python3 -m venv .venv
.venv/bin/pip install tiktoken==0.14.0
.venv/bin/python experiments/tokens/regenerate_fixture.py
git diff -- experiments/tokens/fixture.mjs
```

The generator verifies full-string round trips and asserts that every piece the teaching model treats as one token really is a single `cl100k_base` token. Tests also lock the prompt IDs, subword pieces, staged loop, immutable history, reset, bounded stopping, seed reproducibility, aggregate handling and invalid controls.

## Sources and license

- [OpenAI tiktoken](https://github.com/openai/tiktoken): actual tokenization via the named package and encoding. No tokenizer implementation or full vocabulary is vendored.
- [Transformers LLM tutorial](https://huggingface.co/docs/transformers/main/en/llm_tutorial): tokenization, decoding and generation.
- [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies): greedy selection and sampling.

Original mechanics, examples, preview and tests are MIT licensed under the repository license. No model outputs, weights or paid calls were used.
