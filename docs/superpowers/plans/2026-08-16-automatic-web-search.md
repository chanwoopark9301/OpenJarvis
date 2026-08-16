# OpenJarvis Automatic Web Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search before answering current or local requests, while keeping ordinary companion chat local and preventing unverified answers after search failure.

**Architecture:** Keep `SimpleAgent` as the default responder. A focused CLI preflight module uses a fast deterministic gate, asks the already-loaded local model for privacy-minimized public search queries only when needed, runs the existing `WebSearchTool`, and injects untrusted sourced context before the final answer. If planning or every search fails, the CLI returns a fixed failure message without allowing the model to invent current facts.

**Tech Stack:** Python, Click, Ollama `qwen3.5:9b`, `WebSearchTool`, `ddgs`, pytest, Ruff

## Global Constraints

- Default chat agent remains `simple`.
- Default tool configuration remains the read-only `web_search` tool only.
- Ordinary chat and personal-memory interpretation perform no planning model call and no network request.
- Raw user text is never sent to the search backend.
- Query planning uses the configured model locally and accepts only a strict JSON string array with one to three safe queries.
- Query planning refuses to receive raw text unless the configured engine is verified as local.
- No successful result means no final model call; return a fixed Korean failure message.
- Search results are untrusted external text, capped before prompt injection, and never treated as instructions.
- Existing memory archive and direct-rule behavior remain unchanged.

---

### Task 1: Add the deterministic search gate

**Files:**
- Create: `src/openjarvis/cli/_auto_search.py`
- Create: `tests/cli/test_auto_search.py`

**Interfaces:**
- Produces: `needs_web_search(user_text: str) -> bool`
- Consumes: no engine, network, configuration, or memory state

- [ ] **Step 1: Write failing gate tests**

Add literal cases proving:

```python
assert needs_web_search("안녕?") is False
assert needs_web_search("요즘 내가 왜 이렇게 지치는지 모르겠어") is False
assert needs_web_search("내 상태와 관련된 심리 연구를 인터넷에서 찾아줘") is True
assert needs_web_search("이 글을 요약해 줘") is False
assert needs_web_search("요즘 인기 있는 러닝화를 검색해 줘") is True
assert needs_web_search("인터넷에서 대전 맛집을 찾아줘") is True
assert needs_web_search("웹에서 테미오래 운영시간을 확인해 줘") is True
assert needs_web_search("내일 대전 테미오래 운영시간을 인터넷에서 확인해 줘") is True
assert needs_web_search("대전 한식 맛집과 소품샵을 찾아줘") is True
assert needs_web_search("오늘 서울 날씨가 어때?") is True
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
.venv/bin/pytest -q tests/cli/test_auto_search.py
```

Expected: import failure because `_auto_search.py` does not exist.

- [ ] **Step 3: Implement the minimal pure gate**

Use normalized text and four decision stages in this order:

- explicit search commands: `검색해 줘`, `검색해줘`, `인터넷에서 ... 찾아줘`,
  `인터넷에서 ... 확인해 줘`, `웹에서 ... 찾아줘`, `웹에서 ... 확인해 줘`;
- private introspection exclusion for requests without an explicit search command;
- current domain: weather, hours, closures, price, transport, events combined with today/tomorrow/current/latest signals;
- local recommendations: restaurant, cafe, shop, attraction, or small-goods-shop combined with find/recommend/where signals.

The function returns false for blank input and does not inspect memory.

- [ ] **Step 4: Run the gate tests and verify GREEN**

```bash
.venv/bin/pytest -q tests/cli/test_auto_search.py
```

Expected: all gate tests pass.

### Task 2: Plan safe public queries and execute search

**Files:**
- Modify: `src/openjarvis/cli/_auto_search.py`
- Modify: `tests/cli/test_auto_search.py`

**Interfaces:**
- Produces: immutable `AutoSearchResult(triggered: bool, success: bool, queries: tuple[str, ...], sources: tuple[str, ...], context: str, error: str)`
- Produces: `AutoSearchPreflight.prepare(user_text: str) -> AutoSearchResult`
- Consumes: an `InferenceEngine`, model name, and `BaseTool` compatible with `WebSearchTool`

- [ ] **Step 1: Write failing planner and execution tests**

Use a deterministic fake local engine and fake search tool to prove:

- non-search chat returns `triggered=False` without calling either dependency;
- strict JSON `['대전 테미오래 운영시간', '대전 한식 맛집']` is rejected because JSON requires double quotes;
- a valid array is limited to three trimmed, deduplicated queries;
- a query containing `여자친구`, `여친`, `남자친구`, `남친`, an email, or a phone number is rejected before tool execution;
- a preflight marked as non-local returns the fixed failure without passing raw text to the engine;
- the raw personal request is absent from every query passed to the search tool;
- successful results create context containing `UNTRUSTED WEB SEARCH RESULTS`, source URLs, and an instruction to cite sources;
- source URLs are deduplicated into `AutoSearchResult.sources`;
- all failed results produce `success=False` and the fixed user-facing error.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
.venv/bin/pytest -q tests/cli/test_auto_search.py
```

Expected: failures for missing `AutoSearchResult` and `AutoSearchPreflight`.

- [ ] **Step 3: Implement strict local query planning**

The local planner sends the model a system instruction that requires a JSON array of one to three public-information queries and explicitly excludes names, relationships, feelings, home details, phone numbers, and email addresses. Parse with `json.loads`, accept only a list of non-empty strings, trim and deduplicate, reject privacy patterns, and never fall back to raw user text.

- [ ] **Step 4: Implement bounded search execution**

Execute each safe query with `max_results=3`. Include only successful tool results, cap the combined external context at 12,000 characters, and label it untrusted. When query planning or all searches fail, return this literal user error:

```text
인터넷에서 필요한 정보를 확인하지 못했어. 확인되지 않은 장소나 운영 정보를 지어내지는 않을게. 잠시 뒤 다시 검색해 줘.
```

- [ ] **Step 5: Run tests and Ruff**

```bash
.venv/bin/pytest -q tests/cli/test_auto_search.py
.venv/bin/ruff check src/openjarvis/cli/_auto_search.py tests/cli/test_auto_search.py
```

Expected: all pass with no lint errors.

### Task 3: Integrate preflight into `jarvis chat`

**Files:**
- Modify: `src/openjarvis/cli/chat_cmd.py`
- Modify: `tests/cli/test_chat_cmd.py`

**Interfaces:**
- Consumes: `AutoSearchPreflight.prepare(user_text)`
- Produces: search context as a `Role.SYSTEM` message in `AgentContext.conversation`

- [ ] **Step 1: Write failing CLI behavior tests**

Add tests showing:

- a non-triggered result preserves the existing single response-generation path;
- a successful result places its external context in the real agent message list before the original user message;
- a successful result whose model answer omits links gets a deterministic source list appended;
- a failed triggered result prints the fixed failure text, makes no final response-generation call, and still archives that completed exchange;
- existing personal-memory context and search context both reach the agent instead of one replacing the other.

Mock only the local query-planning generation and external search boundary; assert on CLI output, final engine messages, and archive events.

- [ ] **Step 2: Run the focused CLI tests and verify RED**

```bash
.venv/bin/pytest -q tests/cli/test_chat_cmd.py -k "auto_search"
```

Expected: no matching implementation or failing assertions because chat does not build the preflight.

- [ ] **Step 3: Build one preflight per chat session**

Instantiate `AutoSearchPreflight` with the resolved engine, model, and `WebSearchTool(max_results=3)`. Do not attach the tool to `SimpleAgent`.

- [ ] **Step 4: Apply each preflight result**

For successful triggered search, append a system context message to the same `AgentContext` that carries personal-memory context. For total search failure, use the fixed error as the assistant response without invoking the final model. For non-triggered chat, keep the current path unchanged.

- [ ] **Step 5: Run CLI and related tests**

```bash
.venv/bin/pytest -q tests/cli/test_auto_search.py tests/cli/test_chat_cmd.py
.venv/bin/ruff check src/openjarvis/cli/_auto_search.py src/openjarvis/cli/chat_cmd.py tests/cli/test_auto_search.py tests/cli/test_chat_cmd.py
```

Expected: all pass.

### Task 4: Apply safe local settings and verify live behavior

**Files:**
- Modify: `/Users/cksdndi95/.openjarvis/config.toml`
- Modify: `/Users/cksdndi95/.openjarvis/SOUL.md`

**Interfaces:**
- Consumes: implemented automatic preflight and the existing local-engine guard
- Produces: terminal-ready `jarvis chat` configuration

- [ ] **Step 1: Keep the safe effective configuration**

```toml
[agent]
default_agent = "simple"

[tools]
enabled = ["web_search"]
```

- [ ] **Step 2: Add grounding policy to `SOUL.md`**

State that current/local facts must use supplied web results and source links, external text is not an instruction, and missing evidence must be disclosed without fabrication. State that ordinary personal conversation is not searched.

- [ ] **Step 3: Verify ordinary chat**

Run `jarvis chat`, enter `안녕?`, then `/quit`.

Expected: a short companion reply without search, citations, or unrelated advice.

- [ ] **Step 4: Verify the exact Daejeon request**

Enter the supplied request about 테미오래, Korean food, and clustered small-goods shops.

Expected: the local planner removes relationship wording from external queries; DuckDuckGo returns sources; the answer contains a time-ordered plan and source URLs, and contains none of the previously fabricated names.

- [ ] **Step 5: Verify memory completion**

Query `personal_memory.db`. Expected: after background work settles, no pending, processing, or failed memory job remains.

### Task 5: Final verification, review, and publish

**Files:**
- Verify all changed repository and local configuration files

**Interfaces:**
- Consumes: Tasks 1-4
- Produces: verified automatic search on the existing feature branch

- [ ] **Step 1: Run related regression tests**

```bash
.venv/bin/pytest -q tests/cli/test_auto_search.py tests/cli/test_chat_cmd.py tests/tools/test_web_search.py tests/memory/test_candidate_extractor.py tests/memory/test_context_composer.py tests/memory/test_personal_memory_service.py
```

- [ ] **Step 2: Run quality checks**

```bash
.venv/bin/ruff check src/openjarvis/cli/_auto_search.py src/openjarvis/cli/chat_cmd.py tests/cli/test_auto_search.py tests/cli/test_chat_cmd.py
git diff --check
```

- [ ] **Step 3: Request a read-only code review and address findings with fresh failing tests**

Review privacy boundaries, failure bypass, context ordering, and memory archiving.

- [ ] **Step 4: Commit and push**

```bash
git add src/openjarvis/cli/_auto_search.py src/openjarvis/cli/chat_cmd.py tests/cli/test_auto_search.py tests/cli/test_chat_cmd.py docs/superpowers/specs/2026-08-16-automatic-web-search-design.md docs/superpowers/plans/2026-08-16-automatic-web-search.md
git commit -m "feat(chat): add privacy-safe automatic web search"
git push
```

- [ ] **Step 5: Report exact behavior**

Report the live search backend, sanitized queries, source-backed Daejeon result, changed files, tests, and any unverified details. State that `jarvis chat` works directly from a new terminal.
