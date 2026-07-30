---
name: aoi-verify-push
description: Work on the local AOI_CVbased repository and finish changes through its canonical Todo, validation, intentional staging, commit, and push workflow. Use for general project edits, continuing Todo.md, changing the pipeline/GUI/recipes/tests/docs, synchronizing GitHub, or any AOI task that should leave origin/main updated.
---

# AOI Verify and Push

Use this as the common repository workflow. Let narrower AOI skills provide detector, CUDA, or release-specific steps.

## Workflow

1. Start from the checked-out AOI_CVbased repository root (the directory containing `AGENT.md` and `Todo.md`); do not assume a fixed drive or user profile path.
2. Run `git status --short --branch` and preserve unrelated/user-owned changes.
3. Read `AGENT.md`, then read the relevant sections of the sole canonical `Todo.md`.
4. Implement within the module boundaries defined by `AGENT.md`.
5. Update tests and mark only genuinely completed Todo items; append a dated `完成紀錄` entry.
6. Run every validation required by `AGENT.md` for the changed surface.
7. Stage only explicit task files. Never use `git add .` in a dirty workspace.
8. Inspect the staged diff, commit a concise outcome, and push `main` to `origin/main` unless the user explicitly opts out.
9. Confirm branch synchronization and report validations, commit hash, push result, hardware checks not run, and relevant untracked artifacts.

Do not commit release ZIPs, logs, validation output, generated reports, native build products, or unrelated files.
