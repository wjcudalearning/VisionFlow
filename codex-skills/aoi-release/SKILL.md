---
name: aoi-release
description: Package and publish Windows releases for the local AOI_CVbased project. Use when the user asks to 打包, package, create a ZIP, choose or bump a release version, tag a release, publish GitHub Release assets, reuse a previous release procedure, verify PyInstaller output, or prepare CPU-compatible/CUDA-enabled AOI distribution artifacts.
---

# AOI Release

Separate a local package request from a published release. Do not create tags or GitHub releases unless the user requested publication.

## Choose the scope

- For package-only requests, build and verify the ZIP without tagging or publishing.
- For release requests, require an explicit semantic version. Inspect existing tags and releases before creating anything.
- State whether the package is CPU-compatible only or includes a validated CUDA DLL. If CUDA inclusion is required but the DLL was not built and validated for this commit, stop instead of shipping a stale DLL.
- Treat “same as before” or “以前的方式” as a request to inspect the relevant prior release evidence. Do not infer that it means Chrome, `gh`, or another transport from memory alone. Compare the recent release tag, asset, timestamps, and narrowly relevant prior tool trace when available; do not expose unrelated history or credentials.

## Prepare

1. Read repository `AGENT.md`, release/deployment sections of `Todo.md`, and the README packaging instructions.
2. Inspect `git status`, branch synchronization, current commit, existing tags, existing release assets, and version-bearing UI/docs/configuration.
3. Preserve unrelated ZIPs and artifacts. Never infer the next version from an untracked filename alone.
4. Complete repository tests, compileall, CLI smoke, GUI offscreen smoke, and any detector/CUDA validations required by the release contents.
5. Update version references, Todo, and release-facing documentation consistently, then commit and push before tagging.

## Build and verify

1. Run `build_exe.ps1` from the repository root.
2. Verify `dist\VisionFlow AOI\VisionFlow AOI.exe`, bundled recipes, required Qt/runtime files, and presence/absence of `gpu\visionflow_cuda.dll` according to the intended package.
3. Smoke-test the packaged application on the available machine. Record any GPU/no-GPU matrix that still requires another computer.
4. Create `VisionFlow-AOI-vX.Y.Z-windows-x64.zip` from the whole distribution folder, never the executable alone.
5. Calculate and report SHA-256 plus artifact size. Do not overwrite an existing same-version package without explicit approval.

## Choose the publication transport

Honor an explicit user-selected transport. Otherwise use the first authenticated, non-interactive path already proven for this repository:

1. Existing repository release automation, when it already publishes the requested artifact from the validated tag.
2. Git credential-backed GitHub Releases REST API via `scripts/publish_github_release.ps1`.
3. Chrome only when the task depends on the signed-in browser or the user explicitly selects it.
4. `gh` only when it is authenticated and consistent with the user's request.

For this repository, the proven unattended method is the bundled REST script. It uses the credential already available to `git push`; it does not require `gh auth login` or Chrome file access.

If Chrome returns `Not allowed` while setting a local file, do not repeatedly retry or conclude that GitHub rejected the asset. That error occurs before GitHub receives the file. If the user authorized publication and the REST preflight succeeds, continue with the proven REST method. If the user explicitly required Chrome, report the Chrome file-URL permission blocker instead of silently changing transports.

Never print, persist, or return the credential produced by `git credential fill`. Keep it in memory only and emit sanitized release/asset metadata.

## Publish through the Releases API

Use the bundled script from the repository root:

```powershell
$codexRoot = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE '.codex' }
$publishScript = Join-Path $codexRoot 'skills\aoi-release\scripts\publish_github_release.ps1'
& $publishScript `
  -Repository 'Wwjyun/AOI_CVBased' `
  -Tag 'vX.Y.Z' `
  -AssetPath '.\VisionFlow-AOI-vX.Y.Z-windows-x64.zip' `
  -ReleaseName 'VisionFlow AOI vX.Y.Z' `
  -BodyPath '.\release-notes-vX.Y.Z.md' `
  -ExpectedCommit '<full-commit-sha>' `
  -ExpectedSha256 '<sha256>' `
  -PreflightOnly
```

Review the sanitized preflight result, then rerun without `-PreflightOnly`. The live flow must:

1. Confirm the annotated/lightweight remote tag resolves to the expected commit and that commit is contained in `origin/main`.
2. Refuse an existing same-tag release or ambiguous same-name asset.
3. Create the release as a draft.
4. Upload the ZIP with the in-memory Git credential.
5. Confirm the uploaded byte count, then publish the draft as the latest non-prerelease.
6. Re-fetch the public release, download its canonical asset URL, and verify size and SHA-256.
7. Leave a draft, not an empty public release, if upload or publication fails.

## Final verification

Confirm the release commit is on `origin/main`, the remote tag resolves to it, and the release page shows exactly one expected asset. Independently verify the published asset name, byte count, checksum, recipe count, and CUDA DLL count. Do not treat a successful API response alone as completed publication.

Do not mark untested GPU, no-GPU, stress, or production-recipe acceptance as complete. Report the commit, tag, release URL, direct download URL, asset checksum, CUDA inclusion status, and any remaining cross-machine validation.
