# Context-aware Personal Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a local model interpret memory with recent dialogue while the harness validates evidence, persists typed results, and maintains one canonical dynamic memory source.

**Architecture:** Keep the durable exchange archive and background worker, but give the proposer a bounded recent-dialogue window and schema-constrained output. Remove language-specific meaning extraction, preserve direct model proposals, validate exact user evidence, supersede explicit corrections deterministically by model-assigned subject, and activate canonical prompt projection through a reversible migration gate.

**Tech Stack:** Python 3.11+, SQLite, Ollama JSON-schema output, existing developmental memory pipeline, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-model-led-harness-design.md`

## Global Constraints

- Personal conversation and memory processing remain local-only.
- Raw exchanges are durable before any model call and survive malformed output or shutdown.
- The model owns semantic classification; deterministic code validates structure, provenance, lifecycle, and permissions only.
- Every promoted candidate must carry an exact excerpt from the current user message.
- Direct corrections may supersede older claims; inferred personality or motive claims retain recurrence and confirmation gates.
- Static profile files must not remain a competing dynamic authority after canonical-profile activation.
- Migration is reversible and backs up legacy files before changing prompt authority.

---

### Task 1: Persist exact proposal evidence and bounded dialogue context

**Files:**
- Modify: `src/openjarvis/memory/personal_models.py`
- Modify: `src/openjarvis/memory/archive.py`
- Modify: `tests/memory/test_personal_models.py`
- Modify: `tests/memory/test_personal_memory_archive.py`
- Modify: `tests/memory/test_personal_memory_migration.py`

**Interfaces:**
- Produces: `CandidateDraft.evidence_excerpt: str`, `MemoryCandidate.evidence_excerpt: str`.
- Produces: `PersonalMemoryArchive.recent_exchanges(exchange_id: str, *, limit: int = 6) -> list[ConversationExchange]`.

- [ ] **Step 1: Write failing value-object and archive tests**

```python
def test_candidate_keeps_exact_user_evidence_excerpt():
    draft = CandidateDraft(
        "correction", "The assistant name is 공박사.", 1, 1,
        subject="assistant.name", evidence_excerpt="공박사라고.",
    )
    assert draft.evidence_excerpt == "공박사라고."


def test_recent_exchanges_are_bounded_and_end_before_current(tmp_path):
    archive = seed_three_ordered_exchanges(tmp_path)
    recent = archive.recent_exchanges("exchange-3", limit=2)
    assert [item.id for item in recent] == ["exchange-1", "exchange-2"]
```

- [ ] **Step 2: Run the tests and verify missing fields and method failures**

Run: `.venv/bin/pytest -q tests/memory/test_personal_models.py tests/memory/test_personal_memory_archive.py tests/memory/test_personal_memory_migration.py`

Expected: FAIL on the new constructor argument and archive method.

- [ ] **Step 3: Add schema version 5**

Add `evidence_excerpt TEXT NOT NULL DEFAULT ''` to `memory_candidates`. Update new-database schema, migration code, insert statements, and `_candidate_from_row()`. Keep migration idempotent and verify an existing version 4 database opens without data loss.

- [ ] **Step 4: Add bounded chronological context retrieval**

```python
def recent_exchanges(self, exchange_id: str, *, limit: int = 6):
    with self._lock, self._connect() as connection:
        current = connection.execute(
            "SELECT created_at, session_id, source FROM conversation_exchanges WHERE id = ?",
            (exchange_id,),
        ).fetchone()
        if current is None:
            return []
        scope_column = "session_id" if current["session_id"] else "source"
        scope_value = current[scope_column]
        rows = connection.execute(
            f"SELECT * FROM conversation_exchanges "
            f"WHERE {scope_column} = ? AND created_at < ? "
            f"ORDER BY created_at DESC, id DESC LIMIT ?",
            (scope_value, current["created_at"], max(0, int(limit))),
        ).fetchall()
    return [self._exchange_from_row(row) for row in reversed(rows)]
```

Never include the current exchange or a later exchange.

- [ ] **Step 5: Run archive and migration tests**

Run: `.venv/bin/pytest -q tests/memory/test_personal_models.py tests/memory/test_personal_memory_archive.py tests/memory/test_personal_memory_migration.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/openjarvis/memory/personal_models.py src/openjarvis/memory/archive.py tests/memory/test_personal_models.py tests/memory/test_personal_memory_archive.py tests/memory/test_personal_memory_migration.py
git commit -m "feat(memory): persist exact candidate evidence"
```

### Task 2: Preserve real JSON schemas through the Ollama engine

**Files:**
- Modify: `src/openjarvis/engine/ollama.py`
- Modify: `tests/engine/test_ollama.py`

**Interfaces:**
- Consumes: OpenAI-style `response_format={"type": "json_schema", "json_schema": {"name": str, "schema": dict}}`.
- Produces: Ollama payload `format` containing the inner JSON schema instead of the string `json`.

- [ ] **Step 1: Write a failing payload test**

```python
def test_generate_passes_json_schema_to_ollama(mock_transport):
    schema = {
        "type": "object",
        "properties": {"candidates": {"type": "array"}},
        "required": ["candidates"],
    }
    engine.generate(messages, model="qwen3.5:9b", response_format={
        "type": "json_schema",
        "json_schema": {"name": "memory_candidates", "schema": schema},
    })
    assert mock_transport.last_json["format"] == schema
```

- [ ] **Step 2: Run the test and verify current `format == "json"` failure**

Run: `.venv/bin/pytest -q tests/engine/test_ollama.py::test_generate_passes_json_schema_to_ollama`

Expected: FAIL with `"json" != schema`.

- [ ] **Step 3: Implement exact translation with backward compatibility**

```python
if isinstance(response_format, dict):
    if response_format.get("type") == "json_schema":
        wrapped = response_format.get("json_schema") or {}
        payload["format"] = wrapped.get("schema") or "json"
    else:
        payload["format"] = "json"
```

- [ ] **Step 4: Run all Ollama engine tests**

Run: `.venv/bin/pytest -q tests/engine/test_ollama.py`

Expected: PASS for JSON mode, schemas, tool calls, and control-token filtering.

- [ ] **Step 5: Commit**

```bash
git add src/openjarvis/engine/ollama.py tests/engine/test_ollama.py
git commit -m "feat(ollama): preserve structured output schemas"
```

### Task 3: Replace language rules with a context-aware semantic proposer

**Files:**
- Modify: `src/openjarvis/memory/candidate_extractor.py`
- Modify: `src/openjarvis/memory/personal_service.py`
- Modify: `tests/memory/test_candidate_extractor.py`
- Modify: `tests/memory/test_personal_memory_service.py`

**Interfaces:**
- Consumes: `PersonalCandidateExtractor.extract(exchange, recent_exchanges=())`.
- Produces: schema-constrained `{ "candidates": [...] }` with fields `kind`, `content`, `importance`, `confidence`, `temporal_scope`, `subject`, `target_claim_id`, and `evidence_excerpt`.

- [ ] **Step 1: Replace deterministic-lane tests with context-understanding tests**

```python
def test_bare_name_correction_uses_recent_dialogue():
    engine = FakeEngine({"candidates": [{
        "kind": "correction",
        "content": "The assistant name is 공박사.",
        "importance": 1.0,
        "confidence": 1.0,
        "temporal_scope": "until_changed",
        "subject": "assistant.name",
        "target_claim_id": "",
        "evidence_excerpt": "공박사라고.",
    }]})
    drafts = PersonalCandidateExtractor(engine, "qwen3.5:9b").extract(
        current_exchange("공박사라고.", "네, 공박사입니다."),
        recent_exchanges=(prior_exchange("너의 이름은 조visor가 아니야. 공박사야."),),
    )
    assert drafts[0].content == "The assistant name is 공박사."
    assert drafts[0].subject == "assistant.name"
```

Add tests that reject an evidence excerpt copied only from assistant text, reject an excerpt absent from the current user message, preserve a model-produced constraint, and return an empty list for a greeting.

- [ ] **Step 2: Run focused extractor tests and verify interface failures**

Run: `.venv/bin/pytest -q tests/memory/test_candidate_extractor.py -k "recent_dialogue or evidence_excerpt or model_produced_constraint"`

Expected: FAIL because the extractor has no recent context or evidence field and discards model-produced direct kinds.

- [ ] **Step 3: Define and use one strict output schema**

The schema requires one top-level `candidates` array and rejects extra properties. Pass it through `response_format`. The prompt includes chronological recent exchanges plus the current exchange, marks assistant text as context rather than evidence, and requires every `evidence_excerpt` to be copied exactly from the current user text.

- [ ] **Step 4: Remove semantic regular expressions and direct-kind deletion**

Delete `_deterministic_direct_rules()`, `_classify_behavior_clause()`, durable-rule signal parsing, and this filtering behavior:

```python
drafts = [draft for draft in drafts if draft.kind not in DIRECT_RULE_KINDS]
```

Keep only structural validation, allowed enums, score bounds, maximum lengths, exact evidence containment, and the five-candidate cap.

- [ ] **Step 5: Pass bounded recent exchanges from the worker**

```python
recent = self._archive.recent_exchanges(exchange.id, limit=6)
drafts = self._extractor.extract(exchange, recent_exchanges=tuple(recent))
```

Bump `PERSONAL_CANDIDATE_EXTRACTOR_VERSION` to `personal-memory-v5` so durable older jobs are eligible for reprocessing.

- [ ] **Step 6: Run extractor and worker tests**

Run: `.venv/bin/pytest -q tests/memory/test_candidate_extractor.py tests/memory/test_personal_memory_service.py`

Expected: PASS, including restart recovery and single-worker arbitration.

- [ ] **Step 7: Commit**

```bash
git add src/openjarvis/memory/candidate_extractor.py src/openjarvis/memory/personal_service.py tests/memory/test_candidate_extractor.py tests/memory/test_personal_memory_service.py
git commit -m "refactor(memory): let the local model propose meaning"
```

### Task 4: Validate evidence and supersede explicit corrections

**Files:**
- Modify: `src/openjarvis/memory/evaluator.py`
- Modify: `src/openjarvis/memory/adaptation.py`
- Modify: `src/openjarvis/memory/archive.py`
- Modify: `tests/memory/test_evaluator.py`
- Modify: `tests/memory/test_adaptation.py`
- Modify: `tests/memory/test_context_composer.py`

**Interfaces:**
- Consumes: `MemoryCandidate.evidence_excerpt` and active claims.
- Produces: exact-evidence validation and same-subject correction supersession.

- [ ] **Step 1: Write failing evidence and supersession tests**

```python
def test_evaluator_uses_exact_user_excerpt_not_normalized_claim_text():
    candidate = persist_candidate(
        content="The assistant name is 공박사.",
        evidence_excerpt="공박사라고.",
        user_text="공박사라고.",
    )
    result = MemoryEvaluator(archive).evaluate(candidate.id)
    assert result.applied is True
    assert archive.evidence_texts(result_evidence_ids(archive, candidate.id)) == ("공박사라고.",)


def test_correction_supersedes_active_claim_with_same_subject():
    old = accept_claim(kind="role_preference", subject="assistant.name", content="The assistant name is 조비서.")
    new = evaluate_candidate(kind="correction", subject="assistant.name", content="The assistant name is 공박사.")
    assert archive.get_claim(old.id).state.value == "superseded"
    assert archive.get_claim(new.claim_id).state.value == "active"
```

- [ ] **Step 2: Run focused tests and verify current failures**

Run: `.venv/bin/pytest -q tests/memory/test_evaluator.py tests/memory/test_adaptation.py -k "excerpt or supersede"`

Expected: FAIL because support currently compares normalized claim content and supersession requires an explicit target ID.

- [ ] **Step 3: Validate exact evidence at evaluation time**

Reject when `evidence_excerpt` is empty or not a substring of `exchange.user_text`. Do not compare semantic overlap between normalized claim content and user text. Store the excerpt, not normalized claim content, in `evidence_items.content`.

- [ ] **Step 4: Supersede only model-classified direct corrections with matching subjects**

In `_apply_direct_rule()`, retain explicit `target_claim_id` behavior. When the candidate kind is `CORRECTION` and no target ID is present, select active direct claims whose `subject_scope == candidate.subject`. Return `ACCOMMODATE_SUPERSEDE` for those IDs. Do not use keyword or language matching.

- [ ] **Step 5: Verify composed context contains only the new active correction**

Add a context-composer assertion that a superseded `assistant.name` claim is absent and the new correction appears in direct constraints.

- [ ] **Step 6: Run the developmental memory suite**

Run: `.venv/bin/pytest -q tests/memory/test_evaluator.py tests/memory/test_adaptation.py tests/memory/test_context_composer.py tests/memory/test_developmental_pipeline.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/openjarvis/memory/evaluator.py src/openjarvis/memory/adaptation.py src/openjarvis/memory/archive.py tests/memory/test_evaluator.py tests/memory/test_adaptation.py tests/memory/test_context_composer.py
git commit -m "feat(memory): apply evidence-backed corrections"
```

### Task 5: Establish canonical dynamic prompt memory with reversible migration

**Files:**
- Create: `src/openjarvis/memory/profile_migration.py`
- Modify: `src/openjarvis/core/config.py`
- Modify: `src/openjarvis/cli/memory_cmd.py`
- Modify: `src/openjarvis/cli/chat_cmd.py`
- Modify: `src/openjarvis/memory/context_composer.py`
- Test: `tests/memory/test_profile_migration.py`
- Modify: `tests/cli/test_memory_cmd.py`
- Modify: `tests/cli/test_chat_cmd.py`

**Interfaces:**
- Produces: `LegacyProfileMigrator.stage(user_path: Path, memory_path: Path) -> ProfileMigrationResult`.
- Produces: `activate_canonical_profile(archive, *, backup_paths: tuple[Path, ...]) -> None`.
- Produces: archive metadata key `canonical_profile_active=1`.

- [ ] **Step 1: Write failing reversible-migration tests**

```python
def test_staging_backs_up_profile_files_and_keeps_them_active(tmp_path):
    result = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    assert all(path.exists() for path in result.backup_paths)
    assert archive.get_metadata("canonical_profile_active", "0") == "0"


def test_activation_stops_static_dynamic_profile_injection(tmp_path):
    activate_canonical_profile(archive, backup_paths=backups)
    effective = effective_chat_memory_files(config, original_files, archive)
    assert effective.soul_path == original_files.soul_path
    assert effective.user_path == ""
    assert effective.memory_path == ""
```

- [ ] **Step 2: Run migration tests and verify missing API failures**

Run: `.venv/bin/pytest -q tests/memory/test_profile_migration.py tests/cli/test_memory_cmd.py -k canonical`

Expected: FAIL because canonical-profile staging and activation do not exist.

- [ ] **Step 3: Stage legacy lines as low-trust, non-active candidates**

Back up each existing file with `shutil.copy2`. Record non-empty bullets as `LEGACY_IMPORT` candidates with their file and line provenance. They remain pending and never override direct user evidence automatically.

- [ ] **Step 4: Add explicit activation command and prompt projection gate**

Add `jarvis memory activate-canonical-profile`. It refuses activation without readable backups and a local personal-memory archive. After activation, `chat_cmd` keeps `SOUL.md` but passes empty `USER.md` and `MEMORY.md` paths to `SystemPromptBuilder`; dynamic context comes only from `ContextComposer`.

- [ ] **Step 5: Preserve unresolved pending dialogue across restart**

Extend `ComposedMemoryContext` with `recent_pending_user_messages: tuple[str, ...]`. Populate at most six messages from durable incomplete exchanges, label them as recent dialogue rather than rules, and add them to agent conversation context in chronological user-message order. Do not render assistant text as evidence or a system instruction.

- [ ] **Step 6: Run migration, prompt, and privacy tests**

Run: `.venv/bin/pytest -q tests/memory/test_profile_migration.py tests/memory/test_context_composer.py tests/cli/test_memory_cmd.py tests/cli/test_chat_cmd.py tests/memory/test_developmental_memory_privacy.py`

Expected: PASS; static files remain recoverable, and external providers never receive personal context.

- [ ] **Step 7: Commit**

```bash
git add src/openjarvis/memory/profile_migration.py src/openjarvis/core/config.py src/openjarvis/cli/memory_cmd.py src/openjarvis/cli/chat_cmd.py src/openjarvis/memory/context_composer.py tests/memory/test_profile_migration.py tests/memory/test_context_composer.py tests/cli/test_memory_cmd.py tests/cli/test_chat_cmd.py
git commit -m "feat(memory): make personal memory the canonical profile"
```

### Task 6: Prove restart recall and migrate the current installation

**Files:**
- Create: `tests/memory/test_model_led_memory_acceptance.py`
- Modify: `docs/architecture/personal-memory-harness.md`
- Modify: `docs/architecture/memory.md`

**Interfaces:**
- Consumes: completed Tasks 1-5.
- Produces: end-to-end acceptance coverage and an operational migration record.

- [ ] **Step 1: Add end-to-end acceptance tests**

```python
def test_name_correction_survives_restart(tmp_path):
    first = start_memory_runtime(tmp_path)
    first.complete_turn("공박사라고.", "네, 공박사입니다.", recent=prior_name_turns())
    first.wait_for_candidate_jobs()
    first.stop()

    second = start_memory_runtime(tmp_path)
    context = second.compose("너의 이름이 뭐야?")
    assert "공박사" in context.render()
    assert "조비서" not in context.render()


def test_direct_ban_survives_restart_without_static_profile_file(tmp_path):
    first = start_memory_runtime(tmp_path)
    first.complete_turn(
        "앞으로 공부 이야기를 먼저 꺼내지 마.",
        "알겠어.",
    )
    first.wait_for_candidate_jobs()
    first.stop()
    second = start_memory_runtime(tmp_path, static_profile_files=False)
    assert "공부 이야기를 먼저 꺼내지 마" in second.compose("안녕?").render()


def test_ambiguous_psychology_remains_unpromoted(tmp_path):
    runtime = start_memory_runtime(tmp_path)
    runtime.complete_turn(
        "요즘 내가 왜 그러는지는 나도 잘 모르겠어.",
        "그럴 수 있어.",
    )
    runtime.wait_for_candidate_jobs()
    assert all(
        claim.subject_scope not in {"identity", "mental_health", "relationship_motive"}
        for claim in runtime.archive.get_active_claims()
    )
```

- [ ] **Step 2: Run acceptance and full focused memory suites**

Run: `.venv/bin/pytest -q tests/memory/test_model_led_memory_acceptance.py tests/memory/test_candidate_extractor.py tests/memory/test_personal_memory_service.py tests/memory/test_evaluator.py tests/memory/test_context_composer.py tests/memory/test_profile_migration.py`

Expected: PASS.

- [ ] **Step 3: Run lint and database migration checks**

Run: `.venv/bin/ruff check src/openjarvis/memory src/openjarvis/cli/chat_cmd.py src/openjarvis/cli/memory_cmd.py tests/memory/test_model_led_memory_acceptance.py`

Run: `git diff --check`

Expected: both commands exit successfully.

- [ ] **Step 4: Back up and stage the current profile**

Run: `jarvis memory stage-canonical-profile`

Expected: prints backup paths and imported/skipped counts without printing personal content. Confirm both backups exist before proceeding.

- [ ] **Step 5: Let version 5 reprocess durable name exchanges**

Start `jarvis chat`, wait until pending direct candidate jobs finish, then run the non-content inspection command that reports candidate and job states. Confirm a new active `assistant.name` correction exists and the older conflicting claim is superseded before activation.

- [ ] **Step 6: Activate canonical profile and run the real restart transcript**

Run: `jarvis memory activate-canonical-profile`

Then start a fresh `jarvis chat` and enter:

```text
너의 이름이 뭐야? 기억해봐.
/quit
```

Expected: the answer identifies `공박사`; neither `조비서` nor `조visor` appears; the startup context has no static dynamic-profile conflict.

- [ ] **Step 7: Update documentation and commit**

```bash
git add tests/memory/test_model_led_memory_acceptance.py docs/architecture/personal-memory-harness.md docs/architecture/memory.md
git commit -m "docs: describe context-aware canonical memory"
```
