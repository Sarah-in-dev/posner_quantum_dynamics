# Working in this repo (Posner / quantum dynamics) — read before acting

Every session here follows the same grounding discipline as the rest of Sarah's work. This repo
shares the FULL skill library (symlinked) — the quantum work can draw on the discipline, TALON, and
cross-domain skills, not just the quantum ones.

## Ground before you act
1. **Read `session-discipline` in full** (how to work here; the failure patterns). Always, first.
2. **Read `agent-grounding-protocol`** — run its GROUND sequence: read the relevant skill(s) in full,
   check the most-recent conversation, SHOW what the code does, inspect the data/results.
3. **Identify and read every skill relevant to the task** from the skill index. The full library is
   available: the quantum/Model 6 skills (`model6-*`, `quantum-biology-primer`, `coherence-gated-learning`,
   `entanglement-topology-measurement`, `quantum-computation-and-attribution`, `experiment-design-patterns`),
   the cross-domain bridge (`cross-domain-integration`, `transition-framework`), and the TALON/discipline
   skills. Draw across domains — realizations in one feed the others (`cross-domain-integration`).
4. **Return a `### GROUNDING BRIEF` as your first message** — line-quoted facts tagged by source
   (`[skill X]`, `[recent conversation]`, `[code SHOWN]`, `[data]`) — before writing any code. The human
   scans it and confirms before you build.

Reconciliation rule: **code + data = what IS; skill + recent conversation = what was DECIDED** (and the
locked items you surface but never relitigate). When a file disagrees with prose, the file wins.

## Non-negotiable rules
- **Surgical edits over comprehensive rewrites.** One concrete, validated step at a time.
- **Validate at the data/result level** — "it ran / errors=0" is not validation.
- **Never `datetime.utcnow()`** — use `datetime.now(timezone.utc)`.

## Scope note
This repo does **not** carry the Murmur production gate — no AWS / production-DB access. That gate is
scoped to the Murmur platform repo only. Experiments here run locally or on EC2 per
`model6-codebase-operations` and `experiment-design-patterns`.

## Pointers
- Quantum program map/primer: `quantum-biology-primer`, `model6-architecture`, `model6-codebase-operations`.
- Cross-domain bridge: `cross-domain-integration`. The chat/Code/repo working model: `working-process`.
