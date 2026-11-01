# Third-party source snapshots

The following unmodified JSON files are copied from public upstream repositories
for reproducible architecture and evidence checks. They are not covered by this
repository's MIT license. No model weights or upstream executable code are included.

| Local file (under `experiments/attention-memory/sources/`) | Upstream / attribution | License |
| --- | --- | --- |
| `qwen-config.json` | Qwen / Alibaba Cloud; Copyright 2026 Alibaba Cloud | Apache-2.0; original license and notice preserved in [Qwen-LICENSE.txt](licenses/Qwen-LICENSE.txt) |
| `longbook-config.json` | summerAIO, derived from Qwen/Qwen3.8-27B | Apache-2.0, as declared in the pinned upstream model card; see [Apache-2.0.txt](licenses/Apache-2.0.txt) |
| `longbook-report.json` | summerAIO, derived from Qwen/Qwen3.8-27B | Apache-2.0, as declared in the pinned upstream model card; see [Apache-2.0.txt](licenses/Apache-2.0.txt) |

Exact pinned source URLs, revisions, retrieval date, licensing source URLs and
SHA-256 hashes are recorded in [provenance.json](experiments/attention-memory/sources/provenance.json).
Qwen's pinned revision contains its original LICENSE, copied byte-for-byte here.
The pinned community revision declares `license: apache-2.0` in its model card;
no separate LICENSE or NOTICE file was present in that revision. No additional
upstream NOTICE file was present in the official model's pinned file inventory.
This file documents attribution and provenance; it does not invent an upstream
NOTICE or claim endorsement by either author.

The report contains the upstream author's generic Colab `/content/longbook_next/`
paths, training configuration and author-reported metrics. Those are source
evidence, not results independently reproduced by Learning Machines.

The config's `auto_map` fields are inert text in our JSON checks. Do not load these
snapshots through a model loader or enable upstream remote code as part of this
teaching experiment. The two documented Node commands run only local code and
require no model downloads, dependencies, credentials or network access.
