# OpenJarvis Automatic Web Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `jarvis chat` automatically search for current and local information while keeping ordinary companion conversation direct and private.

**Architecture:** Reuse the existing `OrchestratorAgent` and `WebSearchTool`. The user configuration selects the tool-capable agent and exposes only `web_search`; the persona prompt defines when search is mandatory and forbids invented current facts. Verification uses the real local model and installed DuckDuckGo fallback.

**Tech Stack:** Python, Click, Ollama `qwen3.5:9b`, OpenJarvis `OrchestratorAgent`, `WebSearchTool`, `ddgs`, TOML, Markdown

## Global Constraints

- Automatic search is limited to read-only `web_search` by default.
- Ordinary conversation and personal-memory interpretation do not trigger search.
- Current claims come from search results; failure must not produce invented places or hours.
- Tavily remains optional and DuckDuckGo works without an API key.
- Existing `--tools` overrides and personal-memory behavior remain unchanged.

---

### Task 1: Connect default chat to read-only search

**Files:**
- Modify: `/Users/cksdndi95/.openjarvis/config.toml`
- Verify: `src/openjarvis/cli/chat_cmd.py`
- Verify: `src/openjarvis/cli/_tool_names.py`

**Interfaces:**
- Consumes: `config.agent.default_agent`, `config.tools.enabled`
- Produces: `jarvis chat` with `OrchestratorAgent` and one `WebSearchTool`

- [ ] **Step 1: Capture the current failure**

Run `jarvis chat --model qwen3.5:9b`, then enter `/quit`.

Expected before the change: the banner contains `Agent: simple`.

- [ ] **Step 2: Apply the minimal local configuration**

Preserve unrelated sections and make these values effective:

```toml
[agent]
default_agent = "orchestrator"

[tools]
enabled = ["web_search"]
```

- [ ] **Step 3: Verify configuration and tool resolution**

Run:

```bash
/Users/cksdndi95/.openjarvis/.venv/bin/python -c 'from openjarvis.core.config import load_config; from openjarvis.cli._tool_names import resolve_tool_names; c=load_config(); print(c.agent.default_agent); print(resolve_tool_names(None, c.tools.enabled, c.agent.tools))'
```

Expected literal output:

```text
orchestrator
['web_search']
```

- [ ] **Step 4: Verify the CLI banner**

Run `jarvis chat --model qwen3.5:9b`, then enter `/quit`.

Expected: `Agent: orchestrator`; personal memory still reports direct-rule injection.

### Task 2: Add automatic-search policy

**Files:**
- Modify: `/Users/cksdndi95/.openjarvis/SOUL.md`
- Verify: `src/openjarvis/prompt/builder.py`

**Interfaces:**
- Consumes: `SystemPromptBuilder.build() -> str`
- Produces: a prompt that requires search for current information and forbids fabrication

- [ ] **Step 1: Record the prompt gap**

Build the configured system prompt with `SystemPromptBuilder`.

Expected before the change: no rule requires `web_search` for places, hours, prices, weather, transport, or events.

- [ ] **Step 2: Append the policy to `SOUL.md`**

```markdown
## Internet search

- Use web_search automatically when an answer depends on current or local information, including real places, opening hours, closures, prices, weather, transport, and events.
- Do not use internet search for ordinary conversation, emotional support, personal-memory interpretation, or questions answered from the user's own words.
- Base current and local claims on returned search results and include useful source links.
- If search fails or sources conflict, say what could not be verified. Do not invent places, addresses, hours, prices, or availability.
```

- [ ] **Step 3: Verify prompt composition**

Build the configured system prompt again.

Expected: all four policy rules appear with the existing persona and user profile.

### Task 3: Verify the real search boundary

**Files:**
- Verify: `src/openjarvis/tools/web_search.py`
- Verify: `src/openjarvis/agents/orchestrator.py`
- Verify: `/Users/cksdndi95/.openjarvis/config.toml`
- Verify: `/Users/cksdndi95/.openjarvis/SOUL.md`

**Interfaces:**
- Consumes: `WebSearchTool.execute(query: str, max_results: int = 5) -> ToolResult`
- Produces: sourced current-information answers and direct non-search companion answers

- [ ] **Step 1: Test the free search backend directly**

Run:

```bash
/Users/cksdndi95/.openjarvis/.venv/bin/python -c 'from openjarvis.tools.web_search import WebSearchTool; r=WebSearchTool(max_results=3).execute(query="대전 테미오래 공식 운영시간"); print(r.success); print(r.metadata); print(r.content)'
```

Expected: success is true, metadata names `duckduckgo`, and content includes a source URL. A temporary network failure is reported rather than hidden.

- [ ] **Step 2: Test ordinary companion chat**

Run `jarvis chat`, enter `안녕?`, inspect the reply, then enter `/quit`.

Expected: a short companion reply without citations, invented local facts, or unrelated advice.

- [ ] **Step 3: Test the supplied Daejeon request**

Enter the supplied request about visiting 테미오래 after a 10:00 pickup, Korean food, and a clustered group of small-goods shops.

Expected answer properties:

- time-ordered plan;
- real restaurants and a real shopping area;
- source links;
- verified facts separated from estimates;
- none of the fabricated names from the previous response.

- [ ] **Step 4: Verify memory completion**

Query `personal_memory.db` and group `memory_jobs` by state.

Expected: after the worker finishes, no pending, processing, or failed row remains.

If the live model does not call search for this exact request, stop and report that the
approved configuration-only design is insufficient. Do not silently add a router or a
second model call; that would require a separate design change.

### Task 4: Final verification and handoff

**Files:**
- Verify: `/Users/cksdndi95/.openjarvis/config.toml`
- Verify: `/Users/cksdndi95/.openjarvis/SOUL.md`

**Interfaces:**
- Consumes: completed Tasks 1-3
- Produces: terminal-ready `jarvis chat` with automatic read-only search

- [ ] **Step 1: Run related tests**

```bash
.venv/bin/pytest -q tests/agents/test_orchestrator.py tests/cli/test_chat_cmd.py tests/tools/test_web_search.py tests/memory/test_candidate_extractor.py tests/memory/test_context_composer.py tests/memory/test_personal_memory_service.py
```

- [ ] **Step 2: Run quality checks**

Run Ruff on changed Python files and `git diff --check` on the worktree. Expected: zero errors.

- [ ] **Step 3: Report exact behavior**

Report which search backend served the live test, whether the exact Daejeon request searched, changed files, and any unverified operating details. State that `jarvis chat` can be entered directly in a new terminal.
