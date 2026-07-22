# AGENTS.md — sdk

Read this before touching this repo. Full context lives in
`agentinc/planning/vault/10-packages/sdk.md`.

## The rules

1. Grep `planning/vault/resolver.md` FIRST. Do not scan the tree.
2. When you change public API, update `planning/vault/10-packages/sdk.md`
   and `planning/vault/resolver.md` in a companion PR.
3. Same-turn: never let session knowledge die — add it to the vault.
4. CI enforces via the PR template checklist and (soon) the vault-reminder workflow.

## Cross-package rules

- `sdk/` never imports from platform internals. Platform imports FROM sdk.
  See `planning/vault/20-concepts/open-source-boundary.md`.
- Everything is tenant-scoped. See `planning/vault/20-concepts/multi-tenancy.md`.
