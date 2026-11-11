# Attention and memory

**Teaching illustration — no model is run.**

Run from repository root: `npm test` and `npm run experiment`.

## Assumptions

`growth(n)` describes persistent context-state growth during cached, token-by-token decoding. Each full-attention layer uses `n / 1024` abstract units and each recurrent layer uses one unit. The 64-layer comparisons are full attention (`64n/1024`), Qwen hybrid (`16n/1024 + 48`) and all recurrent (`64`). Equal units are chosen for explanation; slopes and intercepts are not measured memory ratios. Recurrent state exists at zero tokens. The page starts at 1024 tokens and goes to the configured 262144 limit; that limit is not a retrieval result.

This excludes weights, activations, temporary buffers, vision and runtime overhead. Fixed recurrent state does not imply constant total memory, constant processing cost or perfect recall. Full attention does not guarantee exact retrieval.

`illustrate(k)` applies six fixed events to a two-slot summary (`SLOTS`: who, where). The events retell study 01's sentences one piece at a time: the cat on the mat, the king on the throne, the cat on the bed. The full record keeps every event; the summary overwrites each slot's earlier value. `RECALL` is the question both are asked. It is deliberately lossy, not the GDN algorithm or model output. There is no quality score.

## Architecture and evidence

Qwen's official text config has 64 layers: repeating three GatedDeltaNet layers then one full-attention layer. summerAIO's Longbook27-v2 config changes zero-based layers 3, 7, …, 63 to linear attention. Its report records per-layer distillation followed by rank-32 LoRA with `native_lora: true`; the other 48 original GDN layers were also adapted. The UI distinguishes these groups. Conversion is trained work, not a quality-preserving runtime switch.

Text-only source snapshots, exact revisions, retrieval date, URLs and SHA-256 hashes are in `sources/provenance.json`. No custom model code was executed. The community README's hybrid label conflicts with the all-linear config; use the config for layer claims. Configured context is 262144; the report's needle results and needle lengths are empty and its training/cache plan goes up to 16384. There is no demonstrated 262K retrieval result here. Author-reported perplexity is not reproduced or presented as an independent benchmark.

Credit: Qwen team and summerAIO. See the pinned source links in the provenance file.

## Website integration

The site vendors `mechanics.mjs` with an exact Git commit and SHA-256 manifest. From the site checkout run `node scripts/sync-learning-machines.mjs /path/to/learning-machines <full-commit-sha>`. This reads the committed file with `git show`, verifies repository identity and records the pin. Never copy an uncommitted module manually.
