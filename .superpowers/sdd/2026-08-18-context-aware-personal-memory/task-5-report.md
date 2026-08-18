# Task 5 Report: Canonical Dynamic Prompt Memory

## Status

Implemented in the isolated `feature/developmental-personal-model` worktree based on `a1f66f5f36e137b63835ad2a1e82d0c2940c555d`. No staging or activation was run against the user's current installation.

## RED evidence

- Required migration command:
  - `.venv/bin/pytest -q tests/memory/test_profile_migration.py tests/cli/test_memory_cmd.py -k canonical`
  - Failed during collection because `effective_chat_memory_files` and the profile migration API did not exist.
- Pending restart context:
  - `.venv/bin/pytest -q tests/memory/test_context_composer.py tests/cli/test_chat_cmd.py -k 'recent_incomplete or completed_and_empty or pending_memory'`
  - Failed because `ComposedMemoryContext.recent_pending_user_messages` did not exist.
- Named-persona prompt gate:
  - `.venv/bin/pytest -q tests/memory/test_profile_migration.py -k rehydrate`
  - Failed because `SystemPromptBuilder` re-resolved a named persona and restored its USER/MEMORY paths after the initial gate.

## GREEN evidence

- Required migration/context/chat/privacy suite:
  - `.venv/bin/pytest -q tests/memory/test_profile_migration.py tests/memory/test_context_composer.py tests/cli/test_memory_cmd.py tests/cli/test_chat_cmd.py tests/memory/test_developmental_memory_privacy.py`
  - `53 passed`.
- Broader memory/prompt/config/ask regression suite with a disposable product home and the home-path assertion isolated:
  - `403 passed, 14 skipped, 1 deselected`.
  - The separately run named-persona home-path assertion passed (`1 passed`).
- Full CLI sweep reached `471 passed, 2 skipped`; three pre-existing/non-hermetic init download-guidance tests failed because the current hardware/model recommendation did not resolve a model spec, so the download prompt branch was skipped. Task 5 does not touch `init_cmd.py` or those tests.
- Ruff:
  - `ruff check` on every changed source/test file: `All checks passed!`
  - New migration source/test files pass `ruff format --check`.
- `git diff --check`: clean.

## Backup and staging audit

- `LegacyProfileMigrator.stage(user_path, memory_path)` backs up every existing USER/MEMORY file with `shutil.copy2` before importing any line.
- Backup paths are returned in `ProfileMigrationResult.backup_paths` and stored as JSON under `canonical_profile_backup_paths` for the explicit CLI activation step.
- Only non-empty Markdown bullet lines are staged.
- Each line becomes a pending `FACT` candidate with `EvidenceSource.LEGACY_IMPORT`, importance/confidence `0.1`, and `legacy-profile:<file>:<line>` exchange provenance.
- Imported exchanges have empty user and assistant text. Therefore the candidates are not direct evidence, remain non-active after staging, and deterministic evaluation cannot use them to supersede direct user evidence.
- Re-staging the same file/line/content is idempotent for candidate creation while still producing a fresh recovery backup.

## Activation refusal and metadata audit

- `activate_canonical_profile(...)` refuses:
  - an empty backup tuple;
  - missing/non-file/unreadable backup paths;
  - a missing/unreadable personal SQLite archive;
  - a failed SQLite integrity check.
- Validation happens before `canonical_profile_active=1` is stored.
- Activation also persists the validated backup list for recovery/audit.
- CLI commands are explicit:
  - `jarvis memory stage-canonical-profile`
  - `jarvis memory activate-canonical-profile [--backup-path ...]`
- CLI output contains only aggregate backup/candidate counts and activation state; tests verify profile content is never printed.

## Prompt authority audit

- Before activation, staging does not change `canonical_profile_active`, so legacy static profile files remain effective.
- After activation, `effective_chat_memory_files` preserves SOUL and blanks USER/MEMORY.
- Named personas are resolved once before the gate; the returned config clears `persona_name` so the prompt builder cannot rehydrate USER/MEMORY afterward.
- Canonical activation disables the legacy automatic-fact prompt path in chat. Dynamic personal context comes from `ContextComposer` only.
- The activation check reads SQLite metadata through a read-only URI and does not create or migrate the archive on the chat read path.

## Pending-context ordering and authority

- The archive returns the newest six durable incomplete exchanges, then reverses them into chronological order.
- Completed exchanges and empty-user legacy staging records are excluded.
- `ComposedMemoryContext.recent_pending_user_messages` carries only `exchange.user_text`.
- Pending messages are not included by `render()` and therefore never become system rules or evaluated evidence.
- Chat projects them as `Role.USER` messages tagged `personal_memory_context=recent_dialogue`, before live dialogue and without assistant text.

## Privacy audit

- Personal context is allowed only when `is_local_personal_memory_engine` verifies a loopback-local response engine.
- External/unknown providers receive no USER.md, MEMORY.md, legacy facts, canonical claims, or pending user dialogue. SOUL remains as agent identity.
- The external-provider chat test intercepts the final model payload, confirms the private markers are absent, and confirms `compose_configured_personal_context` is never called.
- Existing developmental privacy tests remain green.

## Files

- Added `src/openjarvis/memory/profile_migration.py`.
- Added `tests/memory/test_profile_migration.py`.
- Modified `src/openjarvis/core/config.py`.
- Modified `src/openjarvis/cli/memory_cmd.py`.
- Modified `src/openjarvis/cli/chat_cmd.py`.
- Modified `src/openjarvis/memory/context_composer.py`.
- Modified `src/openjarvis/memory/archive.py` to expose bounded durable incomplete exchanges without leaking SQLite details into the composer.
- Modified `tests/memory/test_context_composer.py`.
- Modified `tests/cli/test_memory_cmd.py`.
- Modified `tests/cli/test_chat_cmd.py`.

## Self-review

- Confirmed the migration is local-only and performs no network calls.
- Confirmed Task 5 never operates on the real USER.md, MEMORY.md, or personal archive.
- Confirmed activation has no implicit fallback that bypasses backup validation.
- Confirmed staging metadata alone cannot disable static files.
- Confirmed prompt gating covers direct chat and agent prompt builders in `chat_cmd`.
- Confirmed pending dialogue is bounded, restart-durable, chronological, user-only, and excluded from system rendering.
- Confirmed external provider payloads exclude every personal-memory path governed by this task.
- Reviewed the final diff for unrelated edits and restored formatter-only changes outside the Task 5 hunks.

## Concerns

- The repository's full CLI suite has three environment-dependent init download-prompt failures described above; they are outside this change and reproduce independently of Task 5 paths.
- Other prompt construction entry points (for example `ask` and server-managed agents) are outside the Task 5 file/interface brief. This task gates `chat_cmd` as specified; extending canonical-profile activation globally should be a separately tested follow-up if desired.
- The worktree remains preserved; no push, merge, or operational activation was performed.

## Fix round 1: trusted activation and shared prompt boundary

### Status

Remediated all nine review findings without operating the user's installed profile or archive. The implementation remains local-only and the staging command remains explicit and reversible.

### RED evidence

- Trusted migration/activation tests initially produced `9 failed, 10 passed, 11 deselected`. Failures demonstrated that an unrelated readable file could activate, changed backups were accepted, imports reread mutable sources, symlinks were accepted, invalid UTF-8 left a partial candidate, no trusted manifest existed, case-only changes collided, activation was unaudited, and the CLI created a missing archive before refusal.
- Prompt/fail-closed/deduplication tests initially produced `4 failed, 1 passed, 52 deselected`. Failures demonstrated that corrupt existing archives restored static profile authority, durable pending messages duplicated live history on direct and agent paths, and `ProactiveAgent` read global USER/MEMORY instead of the shared builder.
- The registered `jarvis chat --agent proactive` external-provider interception test failed separately because the `**kwargs` constructor was not recognized as accepting `prompt_builder`, so SOUL and the gated memory-file configuration never reached the agent.

### GREEN evidence

- Required migration/context/CLI/chat/privacy/proactive suite:
  - `uv run pytest -q tests/memory/test_profile_migration.py tests/memory/test_context_composer.py tests/cli/test_memory_cmd.py tests/cli/test_chat_cmd.py tests/memory/test_developmental_memory_privacy.py tests/agents/test_proactive_agent.py`
  - `75 passed in 0.66s`.
- Broader memory and CLI regression suite:
  - `uv run pytest -q tests/memory tests/cli`
  - `828 passed, 16 skipped, 8 warnings in 12.73s`.
- Full repository stop-on-first-failure run:
  - `5951 passed, 70 skipped, 126 warnings` before one unrelated server-route fixture failure at `tests/server/test_personal_memory_routes.py:43` (`unsupported_by_user_evidence`).
  - The isolated server-route file reproduces as `4 failed`; its fixture creates a direct candidate without the evidence excerpt now required by earlier developmental-memory evaluation work. Task 5 does not modify that route, evaluator, or fixture.
- Ruff check and format check pass on all eleven changed source/test files. `git diff --check` passes.

### Backup, snapshot, and staging guarantees

- Staging rejects source symlinks, backup-directory symlinks, non-files, and pre-existing collision targets.
- Every backup is created under a private archive-adjacent `.canonical-profile-backups` directory, copied with `shutil.copy2`, then restricted to mode `0600`; the directory is restricted to `0700`.
- Candidate extraction reads the exact copied backup bytes. A source mutation immediately after `copy2` cannot change the imported candidate.
- Every staged entry records the configured absolute source path, exact absolute backup path, and SHA-256 of the snapshot in versioned `canonical_profile_staging_manifest` metadata.
- All snapshots are read and UTF-8 decoded before any candidate, exchange, extraction-run, or manifest write. Candidate/exchange rows and manifest publication then occur in one `BEGIN IMMEDIATE` SQLite transaction.
- An invalid second snapshot therefore leaves no partial USER import and does not damage the prior manifest. Filesystem backup artifacts may remain for recovery, but they are not trusted or activatable unless published in the committed manifest.
- Zero-source staging leaves an existing trusted manifest unchanged. Partial staging replaces only entries for sources actually copied and preserves the other trusted recovery entries.
- Provenance identity hashes exact case-sensitive content plus source and line. `US` and `us` now produce distinct legacy candidates.

### Activation refusal, metadata, and audit

- The CLI validates the configured archive through a SQLite `mode=ro` connection before constructing `PersonalMemoryArchive`; a missing archive is refused without creating a file.
- Read-only validation requires a regular non-symlink file, successful SQLite integrity check, and the canonical `archive_metadata` table.
- Activation accepts only the complete ordered backup list recorded in the trusted staging manifest. An unrelated readable file, partial list, reordered list, missing file, symlink, or hash-mismatched snapshot is refused.
- Successful activation atomically records the active flag, activation timestamp, active manifest digest, and an append-only `canonical_profile_activated` decision. Repeated activation remains auditable; it does not silently collapse into an unaudited no-op.
- CLI output remains aggregate-only and never prints staged bullet contents or prompt payloads.

### Prompt authority and privacy audit

- A never-created archive remains inactive for backward compatibility. An existing archive that is unreadable, corrupt, or lacks readable activation metadata fails closed: USER/MEMORY and legacy facts do not regain prompt authority.
- `ProactiveAgent` no longer reads global USER.md or MEMORY.md. Its specialized system prompt calls the common `_apply_persona` boundary.
- Chat constructor inspection now treats an explicit `prompt_builder` parameter or `**kwargs` as prompt-builder-capable, covering the registered proactive class.
- The external-provider `jarvis chat --agent proactive` interception test observes SOUL identity but no USER/MEMORY markers. Canonical personal context and pending dialogue remain local-engine-only.

### Pending dialogue ordering and deduplication

- Equal timestamps are ordered deterministically by exchange ID and returned chronologically.
- Before direct or agent injection, pending user messages are compared with prior live user history using occurrence counts. Only already-represented occurrences are removed, so a legitimate repeated user message retains its remaining multiplicity.
- The current request is excluded from deduplication, preserving a genuinely repeated current turn. Assistant and system content remain excluded from durable pending projection.

### Files changed in fix round 1

- `src/openjarvis/memory/profile_migration.py`
- `src/openjarvis/memory/archive.py`
- `src/openjarvis/core/config.py`
- `src/openjarvis/cli/memory_cmd.py`
- `src/openjarvis/cli/chat_cmd.py`
- `src/openjarvis/agents/proactive_agent.py`
- `tests/memory/test_profile_migration.py`
- `tests/memory/test_context_composer.py`
- `tests/cli/test_memory_cmd.py`
- `tests/cli/test_chat_cmd.py`
- `tests/agents/test_proactive_agent.py`

### Self-review and concerns

- No real installation path was staged or activated; every migration test uses a temporary directory.
- No network provider was called; external privacy tests intercept the model payload.
- The trusted manifest is the sole activation authority. The legacy `canonical_profile_backup_paths` value remains only as compatibility/recovery metadata and is never consulted by activation.
- The only known regression outside the green Task 5 and broader memory/CLI suites is the pre-existing server-route fixture mismatch described above.

## Fix round 2: race-safe snapshots and supplied-archive fail-closed handling

### RED evidence

- `uv run pytest -q tests/memory/test_profile_migration.py -k 'copy_boundary or snapshot_permissions or supplied_archive_sqlite_error or destination_collision'`
  - Initial result: `3 failed, 1 passed, 16 deselected`.
  - The deterministic `copy2` boundary hook replaced the unchecked destination with a symlink, causing the prior implementation to follow it before detecting the unsafe path.
  - A supplied archive raising `sqlite3.DatabaseError` escaped `canonical_profile_is_active` instead of returning the fail-closed state.
- `uv run pytest -q tests/cli/test_chat_cmd.py -k constructor_keyword_audit`
  - Initial result: collection error because no constructor-chain audit existed.
- `uv run pytest -q tests/agents/test_proactive_agent.py -k fresh_process`
  - Initial result: `1 failed, 7 deselected`; importing built-in agents in a fresh process did not register `proactive`, showing that the prior CLI test's direct class import masked the real registry path.

### GREEN evidence

- Focused migration suite: `20 passed in 0.17s`.
- Focused constructor/registration/proactive prompt tests: `4 passed, 24 deselected`.
- Required migration/context/CLI/chat/privacy/proactive suite:
  - `uv run pytest -q tests/memory/test_profile_migration.py tests/memory/test_context_composer.py tests/cli/test_memory_cmd.py tests/cli/test_chat_cmd.py tests/memory/test_developmental_memory_privacy.py tests/agents/test_proactive_agent.py`
  - `80 passed in 0.98s`.
- Broader memory and CLI regression suite:
  - `uv run pytest -q tests/memory tests/cli`
  - `832 passed, 16 skipped, 8 warnings in 12.37s`.
- Ruff check passed after formatting all seven round-two source/test files. `git diff --check` passed.

### Race-safe backup boundary

- The archive-adjacent backup root is atomically created, opened with directory and no-follow flags where supported, verified as a directory through its file descriptor, and forced to mode `0700`.
- Every snapshot receives an atomically created UUID directory and an exclusively reserved `O_EXCL | O_NOFOLLOW` regular file.
- Before `shutil.copy2`, both directory levels are temporarily made non-writable. Root, snapshot-directory, and destination device/inode identities are verified against their open descriptors immediately before the copy.
- `copy2(..., follow_symlinks=False)` retains the required backup metadata semantics. The destination inode is checked again afterward, restricted to `0600`, and fsynced.
- Candidate import bytes are read from a duplicate of the still-open reserved file descriptor, not from a newly resolved path. This closes the post-copy read/symlink race as well as the original check-then-copy race.
- Directory modes are restored to `0700` in `finally` cleanup. Tests assert both directory levels are `0700`, files are `0600`, an injected symlink swap cannot alter its victim, and a deterministic UUID collision cannot overwrite an existing directory.

### Supplied archive fail-closed behavior

- `sqlite3` is now module-scoped in configuration handling, and the supplied-archive metadata branch catches `sqlite3.Error` alongside filesystem/access errors.
- A corrupt or closed supplied archive therefore suppresses USER/MEMORY and legacy fact authority instead of aborting chat or failing open.

### Prompt-builder constructor and registry audit

- The previous broad `**kwargs` check could inject `prompt_builder` into wrappers whose next concrete base rejected it (for example hybrid `LocalCloudAgent` subclasses).
- `_constructor_forwards_keyword` now walks only constructors defined along the class MRO: every `**kwargs` hop must eventually reach an explicit `prompt_builder` receiver before any rejecting signature. This keeps ProactiveAgent and MorningDigestAgent safe while excluding hybrid wrappers that do not forward to a compatible base.
- A focused synthetic-chain test covers safe and unsafe forwarding. The existing registered proactive external-provider interception remains green.
- The audit also found that `openjarvis.agents` omitted the proactive registration in a fresh process. The built-in import list now includes `proactive_agent`, and a subprocess regression proves `jarvis chat --agent proactive` resolves without a test-only pre-import.

### Files changed in fix round 2

- `src/openjarvis/memory/profile_migration.py`
- `src/openjarvis/core/config.py`
- `src/openjarvis/cli/chat_cmd.py`
- `src/openjarvis/agents/__init__.py`
- `tests/memory/test_profile_migration.py`
- `tests/cli/test_chat_cmd.py`
- `tests/agents/test_proactive_agent.py`

### Concerns

- The secure destination implementation uses directory-relative file operations and POSIX permission/inode guarantees available on the supported local Unix execution path. No real installation paths were touched.
- The unrelated server-route fixture mismatch recorded in fix round 1 remains outside Task 5; the required and broader covering suites are green.
