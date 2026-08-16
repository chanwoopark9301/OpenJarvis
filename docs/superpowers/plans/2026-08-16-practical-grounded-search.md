# Practical Grounded Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn partial local web-search evidence into a useful time-ordered answer, retry only missing information once, and remove only unsupported facts instead of discarding the entire answer.

**Architecture:** Keep the existing deterministic search gate and local-only privacy boundary. Add structured search requirements and numbered evidence records, retry unmet requirements once, then require the local response model to return a small itinerary JSON document whose factual fields cite evidence IDs. A deterministic validator removes unsupported fields and renders a natural Korean answer with real source URLs.

**Tech Stack:** Python, Click, Ollama `qwen3.5:9b`, `WebSearchTool`, strict JSON parsing, frozen dataclasses, pytest, Ruff

## Global Constraints

- Default chat agent remains `simple`.
- Ordinary companion chat performs no search-planning, retry, or grounding call.
- Query planning and repair use only the configured locally verified model.
- Raw personal conversation is never passed to the external search tool.
- Relationship, name, phone, email, home, gift, and feeling details remain excluded from public queries.
- Each unmet requirement is retried at most once; no unbounded search loop is allowed.
- Search results remain untrusted reference text and never become instructions.
- A venue name, address, or opening time survives only when its cited evidence contains the same normalized value.
- Unsupported fields are removed per itinerary item; one bad item never discards supported items.
- With no usable evidence, keep the existing fixed search-failure response and skip final answer generation.
- Existing personal-memory archive and direct-rule behavior remain unchanged.

---

### Task 1: Parse numbered evidence records

**Files:**
- Create: `src/openjarvis/cli/_search_evidence.py`
- Create: `tests/cli/test_search_evidence.py`

**Interfaces:**
- Produces: `SearchRequirement(id: str, kind: str, query: str, required_terms: tuple[str, ...])`
- Produces: `EvidenceSource(id: str, requirement_id: str, title: str, url: str, summary: str, source_kind: str)`
- Produces: `parse_search_content(requirement: SearchRequirement, content: str, *, start_index: int) -> tuple[EvidenceSource, ...]`
- Produces: `requirement_is_met(requirement: SearchRequirement, evidence: tuple[EvidenceSource, ...]) -> bool`
- Consumes: the existing `WebSearchTool` text format with `###`, `Source:`, and `Summary:` fields

- [ ] **Step 1: Write failing parser tests**

```python
def test_parse_search_content_assigns_stable_ids_and_fields():
    requirement = SearchRequirement(
        id="R1",
        kind="official",
        query="대전 테미오래 공식 운영시간",
        required_terms=("대전", "테미오래"),
    )
    content = (
        "### 테미오래 공식 누리집\n"
        "Source: https://temiorae.com/\n"
        "Summary: 대전 테미오래 관람 안내와 운영시간"
    )
    evidence = parse_search_content(requirement, content, start_index=4)
    assert [item.id for item in evidence] == ["S4"]
    assert evidence[0].requirement_id == "R1"
    assert evidence[0].url == "https://temiorae.com/"
    assert evidence[0].source_kind == "official"
```

Add separate cases proving malformed blocks and blocks without an HTTP URL are ignored, duplicate URLs are removed, and `requirement_is_met` returns true only when all normalized `required_terms` occur in one evidence record's title, summary, or URL. For an `official` requirement, the matching record must also be classified `official`.

- [ ] **Step 2: Run parser tests and verify RED**

Run: `.venv/bin/pytest -q tests/cli/test_search_evidence.py`

Expected: import failure because `_search_evidence.py` does not exist.

- [ ] **Step 3: Implement immutable evidence types and parser**

```python
@dataclass(frozen=True)
class SearchRequirement:
    id: str
    kind: str
    query: str
    required_terms: tuple[str, ...]

@dataclass(frozen=True)
class EvidenceSource:
    id: str
    requirement_id: str
    title: str
    url: str
    summary: str
    source_kind: str
```

Split only on separator lines, extract the documented fields, and normalize whitespace for matching. Classify source kind with deterministic URL and text signals: `.go.kr`, `공식`, or `누리집` gives `official`; a direct venue or company page gives `business`; known aggregation domains give `listing`; blog and social domains give `review`; everything else gives `other`. Tests must name every signal and expected class instead of making network calls.

- [ ] **Step 4: Run parser tests and Ruff**

```bash
.venv/bin/pytest -q tests/cli/test_search_evidence.py
.venv/bin/ruff check --no-cache src/openjarvis/cli/_search_evidence.py tests/cli/test_search_evidence.py
```

- [ ] **Step 5: Commit Task 1**

```bash
git add src/openjarvis/cli/_search_evidence.py tests/cli/test_search_evidence.py
git commit -m "feat(search): parse numbered web evidence"
```

### Task 2: Plan requirements and retry only missing evidence

**Files:**
- Modify: `src/openjarvis/cli/_auto_search.py`
- Modify: `tests/cli/test_auto_search.py`
- Use: `src/openjarvis/cli/_search_evidence.py`

**Interfaces:**
- Changes: `AutoSearchResult` gains `requirements: tuple[SearchRequirement, ...] = ()` and `evidence: tuple[EvidenceSource, ...] = ()`
- Produces: `AutoSearchPreflight._plan_requirements(user_text: str) -> tuple[SearchRequirement, ...]`
- Produces: `AutoSearchPreflight._plan_retry_query(requirement: SearchRequirement) -> str`
- Consumes: `parse_search_content(...)` and `requirement_is_met(...)` from Task 1

- [ ] **Step 1: Write failing structured-planning tests**

Use a sequenced fake engine. Prove this strict response creates three requirements:

```json
[
  {"id":"R1","kind":"official","query":"대전 테미오래 공식 운영시간 접근","required_terms":["대전","테미오래"]},
  {"id":"R2","kind":"food","query":"대전 테미오래 근처 한식 맛집","required_terms":["대전","한식"]},
  {"id":"R3","kind":"shops","query":"대전 테미오래 주변 소품샵 여러 곳","required_terms":["대전","소품샵"]}
]
```

Assert IDs are unique `R1` through `R3`, `kind` is one of `official`, `food`, `shops`, or `general`, every query passes the privacy filter, and each requirement has one to three non-empty public terms.

- [ ] **Step 2: Run structured-planning tests and verify RED**

Run: `.venv/bin/pytest -q tests/cli/test_auto_search.py -k requirements`

Expected: failure because `requirements` and `_plan_requirements` do not exist.

- [ ] **Step 3: Implement strict requirement planning**

Replace the query-array prompt with the exact object schema above and require an `official` requirement first whenever a visit names a place. Parse only with `json.loads`; never repair permissively and never fall back to raw user text. Keep `AutoSearchResult.queries` populated from `requirement.query` for compatibility.

- [ ] **Step 4: Write failing selective-retry tests**

Make `R1` and `R2` return matching evidence and `R3` return unrelated evidence. Make the local retry planner return `대전 테미오래 인근 소품샵 모음 대흥동 은행동`. Assert:

```python
assert search_tool.queries == [
    "대전 테미오래 공식 운영시간 접근",
    "대전 테미오래 근처 한식 맛집",
    "대전 테미오래 주변 소품샵 여러 곳",
    "대전 테미오래 인근 소품샵 모음 대흥동 은행동",
]
```

Also prove retry failure leaves `R3` unmet without a third search and the retry planner receives only public `R3` fields, never the original personal request.

- [ ] **Step 5: Run selective-retry tests and verify RED**

Run: `.venv/bin/pytest -q tests/cli/test_auto_search.py -k retry`

Expected: failure because unmet requirements are not retried.

- [ ] **Step 6: Implement one retry per unmet requirement**

Execute and parse the first query. If `requirement_is_met` is false, ask the local model for one replacement query using only `id`, `kind`, `query`, and `required_terms`, reject private or duplicate queries, and execute it once. Assign evidence IDs monotonically across all requirements and both passes.

- [ ] **Step 7: Build numbered untrusted context**

Format retained evidence exactly like this and cap the full context at 12,000 characters without renumbering:

```text
[S1] requirement=R1 kind=official
Title: 테미오래 공식 누리집
Source: https://temiorae.com/
Summary: 대전 테미오래 관람 안내와 운영시간
```

Return failure only when no evidence URL exists. Partially met requirements remain successful.

- [ ] **Step 8: Run Task 2 tests and Ruff**

```bash
.venv/bin/pytest -q tests/cli/test_auto_search.py tests/cli/test_search_evidence.py
.venv/bin/ruff check --no-cache src/openjarvis/cli/_auto_search.py src/openjarvis/cli/_search_evidence.py tests/cli/test_auto_search.py tests/cli/test_search_evidence.py
```

- [ ] **Step 9: Commit Task 2**

```bash
git add src/openjarvis/cli/_auto_search.py src/openjarvis/cli/_search_evidence.py tests/cli/test_auto_search.py tests/cli/test_search_evidence.py
git commit -m "feat(search): retry unmet evidence requirements"
```

### Task 3: Validate and render itinerary items independently

**Files:**
- Create: `src/openjarvis/cli/_grounded_answer.py`
- Create: `tests/cli/test_grounded_answer.py`
- Use: `src/openjarvis/cli/_search_evidence.py`

**Interfaces:**
- Produces: `ItineraryItem(time: str, title: str, detail: str, venue: str, address: str, hours: str, source_ids: tuple[str, ...], status: str, check_before_visit: str)`
- Produces: `parse_itinerary_json(content: str) -> tuple[ItineraryItem, ...]`
- Produces: `validate_itinerary(items: tuple[ItineraryItem, ...], evidence: tuple[EvidenceSource, ...]) -> tuple[ItineraryItem, ...]`
- Produces: `render_itinerary(items: tuple[ItineraryItem, ...], evidence: tuple[EvidenceSource, ...]) -> str`
- Produces: `build_itinerary_prompt(user_text: str, context: str) -> str`

- [ ] **Step 1: Write failing strict-parser tests**

Accept only this shape:

```json
{"items":[{"time":"10:30","title":"테미오래 둘러보기","detail":"전시 공간을 여유 있게 둘러봐.","venue":"테미오래","address":"","hours":"","source_ids":["S1"],"status":"confirmed","check_before_visit":"내일 운영 여부 확인"}]}
```

Reject markdown fences, single quotes, a missing `items` list, non-string fields, unknown statuses, and more than eight items. Status values are exactly `confirmed`, `partial`, and `needs_check`.

- [ ] **Step 2: Run parser tests and verify RED**

Run: `.venv/bin/pytest -q tests/cli/test_grounded_answer.py -k parse`

Expected: import failure because `_grounded_answer.py` does not exist.

- [ ] **Step 3: Implement strict parser and immutable item type**

Parse only with `json.loads`, normalize whitespace, deduplicate source IDs while preserving order, and cap every text field at 500 characters.

- [ ] **Step 4: Write failing field-level validation tests**

Use evidence `S1` mentioning `테미오래` and `S2` mentioning a real Korean-food venue. Include another item with unsupported venue `없는식당`, address, and hours. Assert:

```python
assert validated[0].venue == "테미오래"
assert validated[1].venue == ""
assert validated[1].address == ""
assert validated[1].hours == ""
assert validated[1].status == "needs_check"
assert len(validated) == 2
```

Also prove unknown source IDs are removed, an item with no valid source remains as a general plan step, and one unsupported item does not remove a supported item.

- [ ] **Step 5: Run validator tests and verify RED**

Run: `.venv/bin/pytest -q tests/cli/test_grounded_answer.py -k validate`

Expected: failures for missing validation behavior.

- [ ] **Step 6: Implement normalized field matching**

Compare each factual field against the title and summary of only its cited evidence after removing punctuation and whitespace and case-folding Latin text. A `confirmed` status requires either one matching `official` record or two matching records with different URLs; otherwise downgrade it to `partial`. Strip embedded URLs from `detail` and cap it at two sentences. When a factual field is removed, force `status="needs_check"` and add `check_before_visit="방문 전에 최신 정보를 확인해 줘."` when blank.

- [ ] **Step 7: Write failing renderer tests**

Assert the result starts with a practical introduction, preserves item order, shows short confirmation text, omits blank unsupported facts, lists only evidence used by retained items, maps `S1` to its real URL under `확인한 출처`, and never exposes the raw English status values.

- [ ] **Step 8: Implement prompt and renderer**

`build_itinerary_prompt` demands only the strict JSON object, defines statuses, requires source IDs for factual fields, limits output to eight items, and prohibits invented venues, addresses, phones, and hours. `render_itinerary` maps statuses to `확인됨`, `일부 확인`, and `방문 전 확인`.

- [ ] **Step 9: Run Task 3 tests and Ruff**

```bash
.venv/bin/pytest -q tests/cli/test_grounded_answer.py
.venv/bin/ruff check --no-cache src/openjarvis/cli/_grounded_answer.py tests/cli/test_grounded_answer.py
```

- [ ] **Step 10: Commit Task 3**

```bash
git add src/openjarvis/cli/_grounded_answer.py tests/cli/test_grounded_answer.py
git commit -m "feat(search): validate grounded itinerary items"
```

### Task 4: Integrate structured grounding into `jarvis chat`

**Files:**
- Modify: `src/openjarvis/cli/chat_cmd.py`
- Modify: `tests/cli/test_chat_cmd.py`
- Use: `src/openjarvis/cli/_grounded_answer.py`

**Interfaces:**
- Replaces: `_ground_search_answer(content: str, search_result) -> str`
- Produces: `_render_grounded_search_answer(content: str, search_result) -> str | None`
- Produces: `_repair_grounded_json(engine, model: str, content: str, valid_source_ids: tuple[str, ...]) -> str`
- Consumes: `parse_itinerary_json`, `validate_itinerary`, and `render_itinerary`

- [ ] **Step 1: Write the item-level chat regression test**

Replace the current blanket-discard expectation with one supported and one unsupported JSON item. Assert output keeps the supported `테미오래` step, removes the invented venue, retains a practical `방문 전 확인` step, and omits the old whole-answer failure sentence.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/pytest -q tests/cli/test_chat_cmd.py -k "grounded or auto_search"`

Expected: failure because chat still checks for literal source URLs.

- [ ] **Step 3: Integrate parsing, validation, and rendering**

For successful search, add `build_itinerary_prompt(user_input, result.context)` to the same agent context that carries personal memory, run the agent once, parse and validate the JSON, render Korean text, then append and archive only rendered text. Do not change ordinary chat.

- [ ] **Step 4: Write failing one-repair-only tests**

Prove malformed first output triggers exactly one local repair generation using the malformed output, schema instructions, and valid evidence IDs. A valid repair renders normally. A second malformed output uses the safe query-and-source notice. No repair occurs for ordinary chat or valid JSON.

- [ ] **Step 5: Run repair tests and verify RED**

Run: `.venv/bin/pytest -q tests/cli/test_chat_cmd.py -k repair`

Expected: failure because no repair path exists.

- [ ] **Step 6: Implement one local repair attempt**

Call the configured local engine with `temperature=0.0` and `max_tokens=1200`. Require the exact itinerary JSON and only valid evidence IDs. Never send repair content to the web tool or a remote model. Return `None` after the second parse failure so the safe source notice renders.

- [ ] **Step 7: Preserve partial-search and archive behavior**

Add cases proving partially met requirements still call the answer model, all-search failure skips it, memory and grounding contexts both reach the agent, rendered text rather than JSON is archived, and no source ID remains unresolved.

- [ ] **Step 8: Run Task 4 regressions and quality checks**

```bash
.venv/bin/pytest -q tests/cli/test_chat_cmd.py tests/cli/test_auto_search.py tests/cli/test_search_evidence.py tests/cli/test_grounded_answer.py
.venv/bin/ruff check --no-cache src/openjarvis/cli/chat_cmd.py src/openjarvis/cli/_auto_search.py src/openjarvis/cli/_search_evidence.py src/openjarvis/cli/_grounded_answer.py tests/cli/test_chat_cmd.py tests/cli/test_auto_search.py tests/cli/test_search_evidence.py tests/cli/test_grounded_answer.py
git diff --check
```

- [ ] **Step 9: Commit Task 4**

```bash
git add src/openjarvis/cli/chat_cmd.py src/openjarvis/cli/_auto_search.py src/openjarvis/cli/_search_evidence.py src/openjarvis/cli/_grounded_answer.py tests/cli/test_chat_cmd.py tests/cli/test_auto_search.py tests/cli/test_search_evidence.py tests/cli/test_grounded_answer.py
git commit -m "feat(chat): render practical grounded search answers"
```

### Task 5: Live verification and publication

**Files:**
- Verify: `/Users/cksdndi95/.openjarvis/config.toml`
- Verify: `/Users/cksdndi95/.openjarvis/SOUL.md`
- Modify if execution differs: `docs/superpowers/specs/2026-08-16-practical-grounded-search-design.md`
- Modify if execution differs: `docs/superpowers/plans/2026-08-16-practical-grounded-search.md`

**Interfaces:**
- Consumes: Tasks 1 through 4
- Produces: a terminal-ready `jarvis chat` experience on `feature/developmental-personal-model`

- [ ] **Step 1: Verify safe effective settings**

Confirm `default_agent = "simple"`, `enabled = ["web_search"]`, and local grounding rules remain present. Do not broaden the tool list.

- [ ] **Step 2: Run the related suite and quality checks**

```bash
.venv/bin/pytest -q tests/cli/test_auto_search.py tests/cli/test_search_evidence.py tests/cli/test_grounded_answer.py tests/cli/test_chat_cmd.py tests/tools/test_web_search.py tests/memory/test_candidate_extractor.py tests/memory/test_context_composer.py tests/memory/test_personal_memory_service.py
.venv/bin/ruff check --no-cache src/openjarvis/cli/_auto_search.py src/openjarvis/cli/_search_evidence.py src/openjarvis/cli/_grounded_answer.py src/openjarvis/cli/chat_cmd.py tests/cli/test_auto_search.py tests/cli/test_search_evidence.py tests/cli/test_grounded_answer.py tests/cli/test_chat_cmd.py
git diff --check
```

If the three existing live DuckDuckGo tests abort in the sandbox, rerun exactly those three with network permission and report both commands.

- [ ] **Step 3: Verify ordinary and Daejeon chat live**

Run `jarvis chat` with `안녕?`, then the exact request about a 10 o'clock pickup, 테미오래, Korean food, and clustered small-goods shops. Record sanitized initial queries, any retry, evidence IDs and URLs, rendered plan, confirmation labels, and absence of invented factual fields or unresolved IDs.

- [ ] **Step 4: Verify memory completion**

Query `personal_memory.db` and confirm no `pending`, `processing`, or `failed` job remains after background work settles.

- [ ] **Step 5: Run full repository verification**

```bash
.venv/bin/pytest -q --basetemp=/tmp/openjarvis-grounded-search-final -k 'not test_execute_no_api_key and not test_execute_tavily_error and not test_execute_import_error'
```

Run the three excluded live-network tests separately with network permission. If the PostHog exit hook waits after pytest prints its summary, record the printed summary before stopping only that exit hook.

- [ ] **Step 6: Commit corrections and push**

```bash
git status --short
git add src/openjarvis/cli tests/cli
git add -u docs/superpowers/specs/2026-08-16-practical-grounded-search-design.md docs/superpowers/plans/2026-08-16-practical-grounded-search.md
git commit -m "fix(search): finalize practical grounded answers"
git push fork feature/developmental-personal-model
```

- [ ] **Step 7: Report exact behavior and limitations**

Report commit IDs, test counts, live backend, actual retry behavior, final Daejeon output quality, local configuration, and any item still requiring pre-visit confirmation. State that a new terminal can run `jarvis chat` directly.
