# CLAUDE.md — sdk

## Vault first (READ BEFORE ANY WORK)

Canonical knowledge for this repo lives in **`agentinc/planning/vault/`**:

- `resolver.md` / `resolver.csv` — concept → canonical file path. **Grep this first.**
- `10-packages/sdk.md` — this package's public surface, invariants, cross-refs.
- `20-concepts/` — cross-cutting concepts (AgentProtocol, A2A, multi-tenancy, etc.).
- `50-runbooks/` — how to run things locally.

**Every PR that changes this repo MUST also update:**

- `planning/vault/10-packages/sdk.md` if the public surface or invariants moved.
- `planning/vault/resolver.md` if a symbol was added, moved, or removed.
- The relevant `planning/vault/20-concepts/` note if a concept's shape changed.

Open a companion PR in `agentinc/planning`. The PR template below has a checkbox
for this — do not merge without ticking it (or writing why it doesn't apply).

If you learned something non-obvious this session, add it to the vault before
ending. Knowledge that dies with a session forces every future session to
re-derive it.

---

## Security concerns → `70-security/`

If you notice anything security-adjacent while working here — **secret material handled unsafely, missing authz check, injection surface (SQL / shell / prompt), cross-tenant leak, unsafe default, LLM-generated code executed without a sandbox, dependency CVE, rate-limit bypass** — you MUST record it in `planning/vault/70-security/`:

1. Copy `planning/vault/70-security/_template.md` to `YYYY-MM-DD-slug.md`, fill it in.
2. If actionable, also `gh issue create --label security,severity:<lvl>` in this repo and reference the flag.
3. Do NOT silently fix and move on — the board exists so patterns become visible across sessions.

Severity ladder + full process in `planning/vault/70-security/README.md`.

---

