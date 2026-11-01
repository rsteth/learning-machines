# 01 / What is a token?

A small, fixed-fixture walk from text to token IDs, numerical representations, next-token selection, and text again. This is a teaching illustration of a decoder-only language model, **not a model or benchmark**.

![Five stages from text pieces to another piece](preview.svg)

Companion: [What is a token?](https://stethem.com/learning-machines/tokens/) (site release coordinated separately).

## Run

Node.js 18+, no dependencies, network requests or model weights:

```sh
node experiments/tokens/run.mjs
node experiments/tokens/run.mjs --sample
node --test experiments/tokens/mechanics.test.mjs
```

## What is real?

`fixture.mjs` contains actual **tiktoken 0.14.0 / cl100k_base** tokenization, including spaces. It is not Qwen tokenization. For example, `The cat sat on the` has IDs `[791, 8415, 7731, 389, 279]`; `unhappiness` splits into `un`, `h`, `appiness`. IDs are vocabulary labels, not quantities or meanings.

No tokenizer runs in the browser. `decode` only supports the individually UTF-8-decodable tokens recorded in this fixture; it rejects unknown IDs. It is not a general-purpose text encoder or decoder.

All display matrices, six-component vectors, four illustrated layers and probabilities are **invented teaching data**. They are not a trained network, embeddings from a model, measured confidence, or observed outputs. The real process uses learned weights and position information to transform representations, then turns final-position vocabulary scores (logits) into a probability distribution. Generation settings can alter selection; likelihood is not truth confidence.

## Mechanics API

- `createState({mode: 'greedy' | 'sample', seed: uint32})`: start at `tokens` with the fixed prompt.
- `PHASES`: `tokens → embedding → layers → distribution → append`.
- `step(state)`: immutable stage transition. Entering `append` selects and appends one known token. Leaving it starts another cycle, or stops after two appended tokens.
- `distributionFor(state)`: synthetic first-step probabilities (mat 45%, floor 25%, sofa 15%, all other tokens 15%). The second step is explicitly scripted punctuation (`.` at 100%), not a model prediction.
- `chooseToken(distribution, {mode, seed})`: highest individual candidate for this fixture, or deterministic LCG sample (not cryptographic randomness).
- `matricesFor(state)`: deterministic invented input/contextual arrays aligned to the current token count.
- `reset(state)`: replay with the original mode and seed.
- `decode(ids)`: join known fixture pieces, including their leading spaces.

“All other tokens” is an aggregate, **not a token**. Sampling it stops with `outside-fixture`, leaves the prompt unchanged and appends no fake ID. A real model would sample a particular token from its full vocabulary. The ordinary two-token stop is `demo-limit`, not a fabricated end-of-sequence prediction. A real generation loop continues until a stopping rule, token limit or end token; caching can avoid reprocessing all previous tokens. Tokenizers may need to buffer byte fragments before displaying text.

The website owns rendering, controls and editorial copy. It should pin both `mechanics.mjs` and `fixture.mjs` to the same commit and verify SHA-256 hashes. It must keep the synthetic-data labels visible and explain `outside-fixture` without pretending an answer was generated.

## Reproduce the tokenizer fixture

This optional verification installs a tokenizer, **not a model**. Its first encoding load may download a tokenizer vocabulary. Normal demos and tests do not download anything.

```sh
python3 -m venv .venv
.venv/bin/pip install tiktoken==0.14.0
.venv/bin/python experiments/tokens/regenerate_fixture.py
git diff --exit-code -- experiments/tokens/fixture.mjs
```

The generator verifies full-string round trips. Tests also lock the prompt IDs, subword pieces, staged loop, immutable history, reset, bounded stopping, seed reproducibility, aggregate handling and invalid controls.

## Sources and license

- [OpenAI tiktoken](https://github.com/openai/tiktoken): actual tokenization via the named package and encoding. No tokenizer implementation or full vocabulary is vendored.
- [Transformers LLM tutorial](https://huggingface.co/docs/transformers/main/en/llm_tutorial): tokenization, decoding and generation.
- [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies): greedy selection and sampling.

Original mechanics, examples, preview and tests are MIT licensed under the repository license. No model outputs, weights or paid calls were used.
