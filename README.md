# Learning Machines

Small, self-contained experiments for understanding machine learning. Mechanics and evidence live here; the companion website owns the writing, layout and controls.

## Studies

1. [What is a token?](experiments/tokens/README.md) — text, token IDs, a card of notes for each piece and the next piece.
2. [Attention and memory](experiments/attention-memory/README.md) — how much does a model need to keep?

3. [How many dimensions does a movement need?](experiments/manifolds/README.md) — rigid bimanual constraints, cost fields and a fitted skill representation.

Requires Node.js 18 or newer, no dependencies or model downloads.

```sh
npm test
npm run experiment:tokens
npm run experiment
npm run experiment:manifolds
```

These entries are teaching illustrations, not actual model experiments or benchmarks. The token study includes verified tokenizer examples; everything after the token IDs is hand-authored, and each number is computed from the step before it. Future measured experiments should include their dataset, seed, environment and results, and be labelled separately.

Each experiment gets its own folder, runnable entry point, assumptions, sources and tests. No shared framework is required.

## License

Original mechanics, tests, synthetic fixtures and documentation are [MIT licensed](LICENSE). The three copied upstream JSON snapshots retain Apache-2.0 licensing and are explicitly excluded from MIT; see [third-party notices](THIRD_PARTY_NOTICES.md) for attribution, license texts and provenance.
