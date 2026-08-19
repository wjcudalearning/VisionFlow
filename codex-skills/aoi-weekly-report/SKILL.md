---
name: aoi-weekly-report
description: Write or update the AOI_CVBased weekly progress report from repository evidence using the project's Thursday-to-Wednesday reporting cycle. Use for AOI 週報、每週進度、weekly update, or the scheduled Wednesday report; do not use for implementing or validating application code.
---

# AOI Weekly Report

Create a concise, auditable Traditional Chinese weekly report for the `AOI_CVBased` repository.

## Reporting period

- The reporting week is always Thursday 00:00 through Wednesday 23:59 in `Asia/Taipei`, never Monday through Friday.
- For 「這週／本週」, choose the Thursday-to-Wednesday period containing the current date. Save the file as `weekly_reports/WEEKLY_UPDATE_YYYY-MM-DD_to_YYYY-MM-DD.md`.
- When generated before Wednesday ends, keep Wednesday as the period end date but state the actual evidence cutoff time. Do not imply later Wednesday activity was included.
- Match the section style and wording level of the newest existing `weekly_reports/WEEKLY_UPDATE_*.md` unless the user requests another format. If migrating an older repository, move root-level `WEEKLY_UPDATE_*.md` files into `weekly_reports/` before creating the new report.

## Evidence

Use read-only repository evidence:

- Git log bounded to the exact local-time period, including commit date, subject, daily counts, changed files, and aggregate insertions/deletions.
- Dated completed items in `Todo.md`.
- Relevant README, evaluation notes, release notes, and report files changed during the period.
- Current branch, `HEAD`, `origin/main`, tags, and clean/dirty status when useful.

Prefer recorded verification evidence from the commits and `Todo.md`. Phrase it as validation recorded at the related change, not as a fresh rerun.

## No application validation

Weekly-report-only work does not authorize or require `pytest`, test collection, `compileall`, builds, CUDA checks, GUI/CLI smoke tests, packaging, benchmarks, or other application execution. Do not run them. It is acceptable to run the skill validator when the skill itself is being created or changed.

Use only lightweight report checks such as confirming dates, commit totals, Markdown structure, file naming, staged scope, and `git diff --check` limited to the report or skill files. Do not modify `Todo.md`, tag, package, or publish a release unless the user explicitly asks.

## Commit and push

Weekly reporting includes committing and pushing the completed report to `main` → `origin/main` by default. This is a specific exception to any general rule that excludes generated reports from version control.

- Preserve unrelated or user-owned working-tree changes.
- Stage only the new or updated `weekly_reports/` file, intentional weekly-report directory moves, and weekly-report skill documentation changed by the same request. Never use `git add .`.
- Inspect the staged diff and run a staged whitespace check. Do not run application validation.
- Use a concise commit message such as `Add weekly report for YYYY-MM-DD to YYYY-MM-DD` or `Organize weekly reports`.
- Verify the repository and push target, push the current `main` commit to `origin/main`, and confirm local `HEAD` equals `origin/main`.
- Check whether GitHub Actions started for the pushed commit and report its status, but do not replace the skipped local application validation with an extra manual program run.

## Report content

Write an answer-first opening paragraph with the period, main outcomes, commit count, file/change totals, test-suite growth when supported, and latest version/release state when supported.

Then include:

1. One section per day or compact consecutive no-commit range, ordered Thursday through Wednesday.
2. `本週重點整理` for the most important completed outcomes.
3. `版本差異與風險` only for material compatibility, release, evidence, or deployment caveats.
4. `尚待完成` based on still-open `Todo.md` items and limitations explicitly evidenced by the repository.
5. `相關提交` listing every commit in the period in chronological order.

Do not turn commit subjects into unsupported claims. Distinguish code changes, documentation, release preparation, packaged artifacts, and externally published releases. Explicitly note dates with no commits. Do not count the weekly report's own creation as work completed during the period.

## Handoff

Save the Markdown file in the repository's `weekly_reports/` directory, commit and push the scoped report changes, report its absolute path and commit hash, summarize the top outcomes, and state that AOI program validation was intentionally not rerun under this reporting rule.
