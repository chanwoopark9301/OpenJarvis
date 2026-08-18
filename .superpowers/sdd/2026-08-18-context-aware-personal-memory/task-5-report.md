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
