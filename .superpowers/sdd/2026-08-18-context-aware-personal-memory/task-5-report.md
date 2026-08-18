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

## Fix round 3: descriptor-only snapshots and explicit prompt capability

### Controller ruling and status

Replaced the literal `shutil.copy2` implementation with a descriptor-only equivalent. No source or destination content path is re-resolved during copying or import, and unsupported platforms refuse before a backup root or snapshot artifact is created.

### RED evidence

- New descriptor-boundary and failure-invariant selection initially produced `9 failed, 2 passed, 19 deselected`:
  - descriptor-copy/source-swap helpers did not exist;
  - unsupported-platform gating did not exist;
  - injected `fstat` leaked descriptors in the prior helper;
  - injected post-copy chmod left a copied file at source mode `0644`;
  - cleanup and identity failure hooks did not exist.
- The symlink-source regression also exposed an error-contract mismatch after no-follow open correctly refused the link; the safe error was normalized without exposing the configured path.
- The explicit prompt-capability selection initially failed test collection because the capability and lazy-registration APIs did not exist.
- The fresh-process proactive check established the expected side-effect boundary: before an explicit proactive request both proactive agent and proactive tool registration were absent; after the request both were present.
- An injected one-shot directory chmod failure initially left an existing `0777` backup root unchanged, producing a focused RED failure. The mode application now retries once, repairs `0700`, and still preserves the original failure for refusal/audit.

### Descriptor-only source and destination boundary

- `_secure_snapshot_primitives_available` requires POSIX, `O_DIRECTORY`, `O_NOFOLLOW`, `fchmod`, directory-relative `open`/`mkdir`, and fd-capable `utime`. Failure is checked at the beginning of staging, before any backup artifact is created.
- Every source ancestor is opened component-by-component with `O_DIRECTORY | O_NOFOLLOW`. The source itself is opened exactly once with `O_RDONLY | O_NOFOLLOW`, then `fstat` must identify a regular file.
- The backup root and each snapshot directory are created/opened relative to already pinned parent/root descriptors. The destination is reserved with `O_EXCL | O_NOFOLLOW` at mode `0600`.
- `_copy_fd_bytes` uses only `os.read(source_fd)` and `os.write(backup_fd)`. It never receives a filesystem path.
- Source atime/mtime are copied through fd-capable `os.utime`; the destination is fsynced and forced to `0600`. Directories are forced to `0700`.
- Import bytes are read back from the same still-open destination descriptor. A path is opened no-follow only for recovery-path device/inode verification; it is never used as a content source.
- There are no `shutil`, `copy2`, or `_copy2_private_snapshot` references in the migration source or tests.

### Swap and unsupported-platform probes

- Replacing USER.md with an attacker symlink at the copy boundary still backs up/imports the originally opened USER descriptor; attacker content is absent.
- Replacing the entire source ancestor with an attacker directory at the copy boundary likewise preserves the original content.
- Replacing the backup-root ancestor during copying does not write into the attacker tree. Recovery-path verification detects the changed identity and refuses before database publication.
- A simulated unsupported platform raises `secure profile snapshots are not supported` and leaves `.canonical-profile-backups` nonexistent.
- A fixed source mtime regression confirms recovery timestamp preservation.

### Exception and cleanup invariants

- Nested `ExitStack` ownership covers every root, ancestor, source, snapshot, destination, and verification descriptor.
- Cleanup callbacks continue closing remaining descriptors if one cleanup callback raises. `_close_fd` itself retries a failed close once while preserving the failure.
- Injected failures cover copy, recovery-path identity verification, fsync, fstat, chmod, and cleanup. Every case returns `/dev/fd` count to its baseline.
- Surviving backup roots and snapshot directories are always `0700`; destination files are always `0600`, including post-copy failure cases.
- `_force_mode` retries one transient chmod failure before re-raising. This repairs an existing overly broad backup-root mode and preserves the refusal signal.

### Explicit prompt-builder capability and proactive side effects

- Removed all inference that `**kwargs` implies prompt-builder delivery.
- Chat injection now requires both `accepts_prompt_builder=True` and a concrete `prompt_builder` parameter in the selected execution class's effective constructor signature.
- `BaseAgent` declares the capability. ProactiveAgent, OperativeAgent, and MonitorOperativeAgent now expose and forward explicit `prompt_builder` parameters. A swallowing `**kwargs` wrapper is rejected by regression test even when it inherits the capability.
- Proactive is no longer imported by the global built-in-agent import, avoiding its five proactive-tool registrations during ordinary agent loading.
- `_ensure_requested_agent_registered('proactive')` lazily imports and, after registry-reset scenarios, explicitly re-registers only the ProactiveAgent class. A fresh-process test verifies proactive agent/tools are absent before the request and compatible afterward.
- The registered external-provider proactive payload test remains green and excludes USER/MEMORY markers.

### GREEN evidence

- Focused profile migration suite: `31 passed in 0.24s`.
- Focused prompt capability, lazy registration, and proactive privacy selection: `5 passed, 24 deselected in 0.61s`.
- Required Task 5 plus affected MonitorOperative suite:
  - `uv run pytest -q tests/memory/test_profile_migration.py tests/memory/test_context_composer.py tests/cli/test_memory_cmd.py tests/cli/test_chat_cmd.py tests/memory/test_developmental_memory_privacy.py tests/agents/test_proactive_agent.py tests/agents/test_monitor_operative.py`
  - `97 passed in 1.15s`.
- Broader memory and CLI suite:
  - `uv run pytest -q tests/memory tests/cli`
  - `844 passed, 16 skipped, 8 warnings in 12.76s`.
- Full agent suite:
  - `uv run pytest -q tests/agents`
  - `559 passed in 4.33s`.
- Ruff check, Ruff format check, and `git diff --check` pass on all round-three source/test files.

### Files changed in fix round 3

- `src/openjarvis/memory/profile_migration.py`
- `src/openjarvis/cli/chat_cmd.py`
- `src/openjarvis/agents/_stubs.py`
- `src/openjarvis/agents/proactive_agent.py`
- `src/openjarvis/agents/operative.py`
- `src/openjarvis/agents/monitor_operative.py`
- `src/openjarvis/agents/__init__.py`
- `tests/memory/test_profile_migration.py`
- `tests/cli/test_chat_cmd.py`
- `tests/agents/test_proactive_agent.py`

### Concerns

- Secure staging intentionally refuses platforms without the complete no-follow/dir-fd/fd-timestamp primitive set; it does not fall back to a path-based copy.
- The unrelated server-route fixture mismatch recorded in fix round 1 remains outside Task 5.

## Fix round 4: fail-safe permission ordering, descriptor cleanup, and proactive reset

### Controller ruling and status

Kept the descriptor-only snapshot implementation and unsupported-platform preflight from round 3. No literal `copy2` path was restored. This round repairs private modes before fallible identity reads, treats failed `close(2)` calls as single-shot operations, narrows prompt-builder capability to keyword-compatible parameters, and makes explicit proactive loading restore both cached agent and tool registrations.

No real profile, archive, installation, or external provider was changed or invoked.

### RED evidence

- Added the six focused round-four regressions and ran:
  - `uv run pytest -q tests/memory/test_profile_migration.py::test_backup_root_mode_is_repaired_before_root_identity_read tests/memory/test_profile_migration.py::test_destination_mode_is_repaired_before_destination_identity_read tests/memory/test_profile_migration.py::test_failed_close_does_not_retry_a_reused_descriptor tests/cli/test_chat_cmd.py::test_prompt_builder_injection_requires_capability_and_explicit_parameter tests/cli/test_chat_cmd.py::test_requested_proactive_registration_is_lazy_and_compatible tests/agents/test_proactive_agent.py::test_explicit_proactive_request_registers_agent_and_tools_in_fresh_process`
  - Initial result: `5 failed, 1 passed in 0.69s`.
  - The backup-root regression observed the surviving pre-existing root at `0777` after the injected second/root `fstat` failure.
  - The realistic close regression released the first descriptor, reused its number for an unrelated `/dev/null` descriptor, then raised. `_close_fd` retried and closed the unrelated descriptor; the assertion failed with `EBADF`. The same `ExitStack` still closed its other owned descriptor.
  - The prompt regression showed that positional-only `prompt_builder` was incorrectly accepted. The same selection also covers and rejects a `*prompt_builder` variadic positional parameter while retaining ordinary positional-or-keyword and keyword-only receivers.
  - The in-process reset regression imported both proactive modules, cleared `AgentRegistry` and `ToolRegistry`, and requested proactive twice. The agent returned, but `check_permission` (and the other four proactive tools) remained absent because cached decorators did not rerun.
  - The fresh-process absence/request/present regression already passed, isolating the missing behavior to cached-module registry reset.
- The first destination injection replaced `os.open`, which intentionally invalidated the platform capability function's callable-identity check and stopped at the unsupported-platform gate. The harness was corrected without production changes by pinning that already-covered capability check for this failure injection, then rerun:
  - `uv run pytest -q tests/memory/test_profile_migration.py::test_destination_mode_is_repaired_before_destination_identity_read`
  - Result: `1 failed in 0.07s` for the intended reason: the injected destination `fstat` failure left the restricted-umask file at `000` rather than exact `0600`.

### GREEN evidence

- Exact six-test round-four selection: `6 passed in 0.61s`.
- Complete affected migration/chat/proactive files: `63 passed in 1.17s`.
- Required Task 5 plus affected MonitorOperative suite:
  - `uv run pytest -q tests/memory/test_profile_migration.py tests/memory/test_context_composer.py tests/cli/test_memory_cmd.py tests/cli/test_chat_cmd.py tests/memory/test_developmental_memory_privacy.py tests/agents/test_proactive_agent.py tests/agents/test_monitor_operative.py`
  - `100 passed in 1.52s`.
- Broader memory and CLI suite:
  - `uv run pytest -q tests/memory tests/cli`
  - `847 passed, 16 skipped, 8 warnings in 13.57s`; warnings are the existing FastAPI `on_event` deprecations.
  - A final rerun again reached the complete `847 passed, 16 skipped, 8 warnings` summary in `13.86s`, then lingered only in the third-party PostHog client's `atexit` thread join. After the completed test summary was captured, interrupting that stale join exposed the PostHog callback and the test session returned exit code `0`.
- Full agent suite:
  - `uv run pytest -q tests/agents`
  - `559 passed in 4.39s`.
- Post-GREEN test refactor selection: `2 passed in 0.44s`; the destination failure now keys directly off the captured destination descriptor, and the fresh-process test name reflects explicit registration semantics.
- Ruff check and Ruff format check pass on all six round-four source/test files. `git diff --check` passes. The `copy2|shutil|_copy2_private_snapshot` audit has no matches in the migration source or tests.

### Permission ordering and descriptor cleanup

- The safely opened backup root is forced to exact `0700` immediately, before its first `fstat`. An injected second/root `fstat` failure therefore leaves the root private and returns the descriptor count to baseline.
- The exclusively created destination is forced to exact `0600` immediately after `open`, before its first `fstat`. A targeted restrictive-umask plus destination-`fstat` failure leaves the root and snapshot directory at `0700`, the destination at `0600`, and no leaked descriptors.
- `_close_fd` now calls `os.close` exactly once. POSIX permits a failed close to have already released the numeric descriptor, so retrying could close an unrelated descriptor that reused the number.
- `ExitStack` remains the owner of every root-chain, source, backup-root, snapshot, destination, and verification descriptor. Its callback unwinding continues after a close callback raises; the regression confirms the remaining owned descriptor is closed while the reused unrelated descriptor stays open.
- Re-audited all failure paths for copy, recovery-path identity, fsync, source/root/destination/verification `fstat`, chmod, and cleanup. Surviving artifacts retain private modes and the existing failure matrix remains green.

### Prompt capability and proactive lazy registration

- Prompt-builder injection now accepts only `POSITIONAL_OR_KEYWORD` and `KEYWORD_ONLY` parameters named `prompt_builder`, matching the actual keyword call site. Positional-only, variadic positional, variadic keyword wrappers, missing capability flags, and absent parameters are rejected.
- Removed the five proactive tool registration decorators from module import. Importing ordinary built-ins remains free of proactive agent/tool registrations.
- `register_proactive_tools()` explicitly and idempotently registers `check_permission`, `queue_action`, `get_pending_actions`, `record_decision`, and `execute_pending_actions`.
- `_ensure_requested_agent_registered("proactive")` imports the opt-in modules, restores `ProactiveAgent` if the agent registry was cleared, and invokes explicit tool registration on every request. A second request is a no-op for already-correct registrations.
- The in-process regression exercises cached modules after both registries are cleared and checks the exact agent class plus all five tool classes. The subprocess regression verifies absence before an explicit request and complete presence after the first and second requests.
- `ProactiveAgent` continues constructing its concrete proactive tools directly, so removing registry decorators does not alter its tool executor contents. No import cycle or unrelated global registration was introduced.

### Files changed in fix round 4

- `src/openjarvis/memory/profile_migration.py`
- `src/openjarvis/cli/chat_cmd.py`
- `src/openjarvis/tools/proactive_tools.py`
- `tests/memory/test_profile_migration.py`
- `tests/cli/test_chat_cmd.py`
- `tests/agents/test_proactive_agent.py`
- `.superpowers/sdd/2026-08-18-context-aware-personal-memory/task-5-report.md`

### Self-review and concerns

- The unsupported-platform gate still runs before backup-root or snapshot creation; descriptor-only copying remains the only staging implementation.
- A failed `close` is intentionally not retried because the descriptor's post-error state is unspecified. The original cleanup error is propagated while independent stack callbacks continue.
- Explicit proactive registration does not replace an unrelated pre-existing registry entry with the same key; normal empty/reset and repeated-request paths register the intended classes idempotently.
- No literal `copy2`, path-based content reread, real migration, installation write, push, merge, or external request was performed.
- The unrelated server-route fixture mismatch recorded in fix round 1 was not part of the requested round-four suites and remains outside Task 5.
