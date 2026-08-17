# General Life Information Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the itinerary-only search path with one generic local planner and evidence-grounded answer path that handles unfamiliar life-information requests without per-domain models.

**Architecture:** A fast capability gate leaves ordinary conversation unchanged and sends search, comparison, planning, or external-action requests to a local `RequestPlanner`. An `AssistantPreflight` either asks one clarifying question, reports an unavailable action, or executes up to three privacy-checked public searches. Search answers use a generic block document with exact evidence excerpts, deterministic validation, and four presentation styles instead of the itinerary schema.

**Tech Stack:** Python 3.13, dataclasses, existing OpenJarvis engine and `WebSearchTool`, Click and Rich, pytest, Ruff.

## Global Constraints

- Do not create weather, exercise, product, news, restaurant, or travel-specific models or handlers.
- Clear ordinary conversation must keep the existing single-generation path.
- All planning and repair calls must use the configured local personal-memory engine.
- Send only validated public queries to the external search tool; never send raw private conversation context.
- Ask at most one clarifying question per turn and do not ask for information already present in recent conversation.
- Execute at most three initial searches and one retry per unmet requirement.
- Never claim that an external action completed without a tool execution record.
- Every factual answer block must cite an existing source and an exact excerpt from that source.
- Keep current personal-memory recording and completed-exchange publication behavior unchanged.
- Browser interaction, ordering, checkout, booking, and other external writes remain out of scope.

---

### Task 1: Generic request planning contract

**Files:**
- Create: `src/openjarvis/cli/_request_plan.py`
- Create: `tests/cli/test_request_plan.py`

**Interfaces:**
- Consumes: local engine `generate(messages, model=..., temperature=..., max_tokens=...)`; recent `Message` objects.
- Produces: `PlanRequirement(id: str, query: str, required_terms: tuple[str, ...])`.
- Produces: `RequestPlan(mode: str, goal: str, clarifying_question: str, response_style: str, requirements: tuple[PlanRequirement, ...])`.
- Produces: `needs_assistant_planning(user_text: str, pending_clarification: bool = False) -> bool`.
- Produces: `RequestPlanner.plan(user_text: str, recent_messages: tuple[Message, ...]) -> RequestPlan | None`.

- [ ] **Step 1: Write failing capability-gate tests**

```python
@pytest.mark.parametrize("text", ["안녕?", "오늘 좀 피곤했어", "그냥 이야기하자"])
def test_clear_companion_chat_skips_planner(text):
    assert needs_assistant_planning(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "오늘 날씨 검색해 볼래?",
        "운동법을 인터넷에서 찾아줘",
        "무설탕 음료 세 개를 비교해 줘",
        "대전 데이트 계획을 구체화해 줘",
        "쿠팡에서 음료수를 주문해 줘",
    ],
)
def test_information_and_action_requests_use_planner(text):
    assert needs_assistant_planning(text) is True


def test_short_follow_up_after_work_request_uses_planner():
    assert needs_assistant_planning(
        "경기도 과천시야.", pending_clarification=True
    ) is True
```

- [ ] **Step 2: Run the gate tests and verify the missing module failure**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_request_plan.py -k 'planner'`

Expected: collection fails because `openjarvis.cli._request_plan` does not exist.

- [ ] **Step 3: Add the minimal capability gate**

```python
_WORK_SIGNALS = (
    "검색", "찾아", "확인", "추천", "비교", "계획", "구체화",
    "주문", "구매", "예약", "보내", "결제",
)


def needs_assistant_planning(
    user_text: str,
    pending_clarification: bool = False,
) -> bool:
    normalized = re.sub(r"\s+", "", user_text.casefold())
    direct_work = any(signal in normalized for signal in _WORK_SIGNALS) or (
        any(word in normalized for word in ("오늘", "내일", "현재", "최신"))
        and any(word in normalized for word in ("날씨", "가격", "시간", "재고"))
    )
    return direct_work or pending_clarification
```

- [ ] **Step 4: Run the gate tests and verify they pass**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_request_plan.py -k 'planner'`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing strict-plan tests**

```python
def test_weather_without_location_returns_one_clarifying_question():
    engine = FakeEngine(
        '{"mode":"clarify","goal":"오늘 날씨 알려주기",'
        '"clarifying_question":"어느 지역의 날씨를 볼까?",'
        '"response_style":"direct","requirements":[]}'
    )
    plan = RequestPlanner(engine, "test-model").plan(
        "오늘 날씨 검색해 볼래?", ()
    )
    assert plan.mode == "clarify"
    assert plan.clarifying_question == "어느 지역의 날씨를 볼까?"
    assert plan.requirements == ()


def test_recent_public_context_resolves_follow_up_without_reasking():
    engine = FakeEngine(
        '{"mode":"search","goal":"과천시 오늘 날씨",'
        '"clarifying_question":"","response_style":"direct",'
        '"requirements":[{"id":"R1","query":"경기도 과천시 오늘 날씨",'
        '"required_terms":["과천시","날씨"]}]}'
    )
    history = (Message(role=Role.USER, content="나는 과천시에 있어."),)
    plan = RequestPlanner(engine, "test-model").plan("그럼 오늘은 어때?", history)
    assert plan.mode == "search"
    assert plan.requirements[0].query == "경기도 과천시 오늘 날씨"


def test_action_plan_cannot_contain_search_requirements():
    engine = FakeEngine(
        '{"mode":"action","goal":"음료수 주문",'
        '"clarifying_question":"","response_style":"direct",'
        '"requirements":[{"id":"R1","query":"음료수",'
        '"required_terms":["음료수"]}]}'
    )
    assert RequestPlanner(engine, "test-model").plan("음료수 주문해 줘", ()) is None
```

- [ ] **Step 6: Implement strict plan parsing, bounded recent context, and one repair**

Use these exact contracts:

```python
_MODES = frozenset({"chat", "clarify", "search", "action"})
_STYLES = frozenset({"direct", "list", "comparison", "steps"})

@dataclass(frozen=True)
class PlanRequirement:
    id: str
    query: str
    required_terms: tuple[str, ...]

@dataclass(frozen=True)
class RequestPlan:
    mode: str
    goal: str
    clarifying_question: str
    response_style: str
    requirements: tuple[PlanRequirement, ...]
```

Pass at most the latest six non-system messages to the local model. Validate exact fields, sequential IDs, one to three search requirements, nonempty public terms, no relationship words, no contact details, and no ideographs. `clarify` requires one question and no requirements; `action` and `chat` require no requirements. If the first JSON document is malformed, issue exactly one local repair call containing only the malformed document and schema; return `None` if repair also fails.

- [ ] **Step 7: Run all request-plan tests**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_request_plan.py`

Expected: all tests pass.

- [ ] **Step 8: Commit the planner**

```bash
git add src/openjarvis/cli/_request_plan.py tests/cli/test_request_plan.py
git commit -m "feat(chat): add generic local request planner"
```

### Task 2: Generic assistant preflight and public search execution

**Files:**
- Create: `src/openjarvis/cli/_assistant_preflight.py`
- Create: `tests/cli/test_assistant_preflight.py`
- Modify: `src/openjarvis/cli/_search_evidence.py`
- Modify: `tests/cli/test_search_evidence.py`

**Interfaces:**
- Consumes: `RequestPlanner.plan(...)`, `WebSearchTool.execute(...)`, and existing evidence parsing.
- Produces: `AssistantPreflightResult(triggered, mode, response_style, goal, clarifying_question, success, queries, evidence, context, error, request_text)`.
- Produces: `AssistantPreflight.prepare(user_text: str, recent_messages: tuple[Message, ...] = (), pending_clarification: bool = False) -> AssistantPreflightResult`.

- [ ] **Step 1: Write failing mode-routing tests**

```python
def test_clear_chat_does_not_call_planner_or_search():
    result = AssistantPreflight(ExplodingPlanner(), ExplodingSearchTool()).prepare(
        "안녕?", ()
    )
    assert result.triggered is False


def test_planner_chat_mode_returns_to_normal_generation():
    plan = RequestPlan("chat", "사용자와 대화", "", "direct", ())
    result = AssistantPreflight(FixedPlanner(plan), ExplodingSearchTool()).prepare(
        "그건 어때?", ()
    )
    assert result.triggered is False


def test_clarify_returns_question_without_search():
    plan = RequestPlan("clarify", "오늘 날씨", "어느 지역의 날씨를 볼까?", "direct", ())
    result = AssistantPreflight(FixedPlanner(plan), ExplodingSearchTool()).prepare(
        "오늘 날씨 검색해 줘", ()
    )
    assert result.mode == "clarify"
    assert result.clarifying_question == "어느 지역의 날씨를 볼까?"


def test_action_returns_unavailable_without_claiming_completion():
    plan = RequestPlan("action", "음료수 주문", "", "direct", ())
    result = AssistantPreflight(FixedPlanner(plan), ExplodingSearchTool()).prepare(
        "쿠팡에서 음료수 주문해 줘", ()
    )
    assert "직접 조작하는 기능은 아직 연결되지 않았어" in result.error
    assert "주문했어" not in result.error
```

- [ ] **Step 2: Run the routing tests and verify the missing module failure**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_assistant_preflight.py -k 'chat or clarify or action'`

Expected: collection fails because `openjarvis.cli._assistant_preflight` does not exist.

- [ ] **Step 3: Implement the result object and non-search modes**

```python
@dataclass(frozen=True)
class AssistantPreflightResult:
    triggered: bool
    mode: str = "chat"
    response_style: str = "direct"
    goal: str = ""
    clarifying_question: str = ""
    success: bool = False
    queries: tuple[str, ...] = ()
    evidence: tuple[EvidenceSource, ...] = ()
    context: str = ""
    error: str = ""
    request_text: str = ""
```

For planner mode `chat`, return `triggered=False` so the existing normal generation runs. For `clarify`, return the validated question without search. For `action`, return a fixed unavailable-action message. For a failed plan, return `요청을 이해하지 못했어. 찾고 싶은 내용을 한 문장으로 다시 말해 줄래?` without search.

- [ ] **Step 4: Run routing tests and verify they pass**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_assistant_preflight.py -k 'chat or clarify or action'`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing generic search and privacy tests**

```python
def test_search_executes_generic_requirements_without_domain_kinds():
    plan = RequestPlan(
        "search", "과천시 오늘 날씨", "", "direct",
        (PlanRequirement("R1", "경기도 과천시 오늘 날씨", ("과천시", "날씨")),),
    )
    result = AssistantPreflight(FixedPlanner(plan), WeatherSearchTool()).prepare(
        "과천시 오늘 날씨를 검색해 줘", ()
    )
    assert result.success is True
    assert result.response_style == "direct"
    assert result.queries == ("경기도 과천시 오늘 날씨",)


def test_private_relationship_context_never_reaches_search_tool():
    history = (Message(role=Role.USER, content="여친과 과천에 갈 거야."),)
    plan = RequestPlan(
        "search", "과천 날씨", "", "direct",
        (PlanRequirement("R1", "여친 과천 날씨", ("과천", "날씨")),),
    )
    result = AssistantPreflight(FixedPlanner(plan), RecordingSearchTool()).prepare(
        "날씨를 찾아줘", history
    )
    assert result.success is False
    assert result.queries == ()
```

- [ ] **Step 6: Implement search, relevance filtering, retry, and context**

Reuse numbered evidence parsing while removing `SearchRequirement.kind` and the `official`, `food`, and `shops` planning categories. Make evidence parsing and requirement checks consume `PlanRequirement` from `_request_plan.py`. Relevance uses required terms and place anchors generically. Keep one retry only for each unmet requirement, URL deduplication, social-title checks, the existing context limit, and an explicit untrusted-evidence warning.

- [ ] **Step 7: Run preflight and evidence tests**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_assistant_preflight.py tests/cli/test_search_evidence.py`

Expected: all tests pass.

- [ ] **Step 8: Commit generic preflight**

```bash
git add src/openjarvis/cli/_assistant_preflight.py src/openjarvis/cli/_search_evidence.py tests/cli/test_assistant_preflight.py tests/cli/test_search_evidence.py
git commit -m "feat(search): add generic assistant preflight"
```

### Task 3: Generic grounded response document

**Files:**
- Create: `src/openjarvis/cli/_grounded_response.py`
- Create: `tests/cli/test_grounded_response.py`

**Interfaces:**
- Produces: `Support(source_id: str, excerpt: str)`.
- Produces: `ResponseBlock(kind: str, text: str, supports: tuple[Support, ...])`.
- Produces: `GroundedResponse(lead: str, blocks: tuple[ResponseBlock, ...], follow_up: str)`.
- Produces: `build_grounded_prompt(...)`, `parse_grounded_json(...)`, `validate_grounded_response(...)`, `render_grounded_response(...)`, and `repair_grounded_json(...)`.

- [ ] **Step 1: Write failing strict-parser tests**

```python
def test_parse_accepts_only_generic_block_schema():
    content = (
        '{"lead":"과천시 날씨를 확인했어.","blocks":['
        '{"kind":"fact","text":"현재 기온은 24도야.",'
        '"supports":[{"source_id":"S1","excerpt":"현재 기온 24도"}]},'
        '{"kind":"advice","text":"늦게 나가면 얇은 겉옷을 챙겨.",'
        '"supports":[]}],"follow_up":""}'
    )
    result = parse_grounded_json(content)
    assert result.blocks[0].kind == "fact"
    assert result.blocks[0].supports[0].source_id == "S1"


@pytest.mark.parametrize("content", ["```json\n{}\n```", "{'lead':'x'}", '{"items":[]}'])
def test_parse_rejects_non_strict_documents(content):
    with pytest.raises(ValueError):
        parse_grounded_json(content)
```

- [ ] **Step 2: Run parser tests and verify the missing module failure**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_grounded_response.py -k 'parse'`

Expected: collection fails because `openjarvis.cli._grounded_response` does not exist.

- [ ] **Step 3: Implement dataclasses and strict parsing**

Accept only `fact`, `advice`, and `uncertain`; at most eight blocks; at most three supports per block; exact fields only; strings capped at 500 characters; source IDs deduplicated; ideographs removed from user-visible fields.

- [ ] **Step 4: Run parser tests and verify they pass**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_grounded_response.py -k 'parse'`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing validation tests**

```python
def test_fact_requires_exact_excerpt_and_matching_number():
    evidence = (
        EvidenceSource("S1", "R1", "과천 날씨", "https://weather.test", "현재 기온 24도", "official"),
    )
    response = GroundedResponse(
        "날씨를 확인했어.",
        (
            ResponseBlock("fact", "현재 기온은 29도야.", (Support("S1", "현재 기온 24도"),)),
            ResponseBlock("fact", "현재 기온은 24도야.", (Support("S1", "현재 기온 24도"),)),
        ),
        "",
    )
    validated = validate_grounded_response(response, evidence)
    assert [block.text for block in validated.blocks] == ["현재 기온은 24도야."]


def test_one_bad_fact_does_not_remove_supported_fact_or_advice():
    evidence = (
        EvidenceSource("S1", "R1", "제품 안내", "https://product.test", "용량 500밀리리터", "business"),
    )
    response = GroundedResponse(
        "비교했어.",
        (
            ResponseBlock("fact", "가격은 1000원이야.", (Support("S404", "가격 1000원"),)),
            ResponseBlock("fact", "용량은 500밀리리터야.", (Support("S1", "용량 500밀리리터"),)),
            ResponseBlock("advice", "취향에 맞는 제품을 골라.", ()),
        ),
        "",
    )
    validated = validate_grounded_response(response, evidence)
    assert len(validated.blocks) == 2
```

- [ ] **Step 6: Implement deterministic evidence validation**

Require each fact support ID to exist and each excerpt to be an exact normalized substring of that source title or summary. Every number, clock, currency value, and explicit address fragment in fact text must occur in the combined excerpts. Advice has no support and cannot contain URLs, prices, clock values, addresses, phone numbers, or unsupported completion claims. A failing uncertainty block remains source-free uncertainty rather than becoming a fact.

- [ ] **Step 7: Write failing style-rendering tests**

```python
@pytest.mark.parametrize("style", ["direct", "list", "comparison", "steps"])
def test_each_style_renders_generic_blocks_and_only_used_sources(style):
    rendered = render_grounded_response(validated_fixture(), evidence_fixture(), style)
    assert "https://used.test" in rendered
    assert "https://unused.test" not in rendered
    assert "방문 전 확인" not in rendered
    assert "일정으로 정리" not in rendered
```

- [ ] **Step 8: Implement prompt, four renderers, and one repair**

The prompt includes the exact schema, original request, goal, selected style, and untrusted evidence. Render direct paragraphs, bullet lists, comparison bullets, or numbered steps without domain-specific language. `repair_grounded_json` makes exactly one local call, restricts source IDs to the evidence set, and forbids new facts.

- [ ] **Step 9: Run all grounded-response tests**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_grounded_response.py`

Expected: all tests pass.

- [ ] **Step 10: Commit generic grounded responses**

```bash
git add src/openjarvis/cli/_grounded_response.py tests/cli/test_grounded_response.py
git commit -m "feat(search): add generic grounded responses"
```

### Task 4: Connect generic planning to `jarvis chat`

**Files:**
- Modify: `src/openjarvis/cli/chat_cmd.py`
- Modify: `tests/cli/test_chat_cmd.py`

**Interfaces:**
- Consumes: `AssistantPreflight.prepare(user_text, recent_messages)` and generic grounded response functions.
- Produces: user-visible behavior for chat, clarify, action, search success, malformed answer, and search failure.

- [ ] **Step 1: Replace itinerary fixtures with failing generic chat tests**

```python
def test_weather_search_uses_direct_grounded_answer_not_itinerary_copy():
    result = invoke_chat("과천시 오늘 날씨를 검색해 줘")
    assert "현재 기온은 24도야" in result.output
    assert "방문 전 확인" not in result.output
    assert "일정으로 정리" not in result.output


def test_missing_location_prints_clarifying_question_without_generation():
    preflight = AssistantPreflightResult(
        True, mode="clarify", clarifying_question="어느 지역의 날씨를 볼까?"
    )
    result = invoke_chat_with_preflight("오늘 날씨를 검색해 줘", preflight)
    assert "어느 지역의 날씨를 볼까?" in result.output
    assert engine.generate.call_count == 0


def test_recent_messages_are_available_to_follow_up_planner():
    invoke_two_turn_chat("나는 과천시에 있어.\n그럼 오늘 날씨는 어때?\n/quit\n")
    assert preflight.received_messages[-1].content == "나는 과천시에 있어."
```

- [ ] **Step 2: Run new chat tests and verify itinerary behavior fails them**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_chat_cmd.py -k 'weather or clarifying or recent_messages'`

Expected: tests fail because chat still builds itinerary JSON or does not pass history.

- [ ] **Step 3: Replace chat orchestration**

Rename the builder to `_build_assistant_preflight`. Keep a session-local `pending_clarification` flag. Call `prepare(user_input, tuple(history[:-1][-6:]), pending_clarification=pending_clarification)`, set the flag only when the result mode is `clarify`, and clear it after the next planned turn. Handle `clarify` with the question, `action` with the fixed unavailable message, failed search with its error, and successful search with the generic grounded prompt. Parse and validate once, repair malformed JSON once, then use a generic safe notice saying `사실을 출처와 연결하지 못했어` instead of mentioning places or times.

- [ ] **Step 4: Preserve memory and history behavior**

Clarify, action, failed search, and successful search responses each append one assistant message and reach `record_and_publish_completed_exchange` exactly once. Ordinary chat remains one model generation and does not invoke the request planner.

- [ ] **Step 5: Run all chat tests**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_chat_cmd.py`

Expected: all tests pass.

- [ ] **Step 6: Commit chat integration**

```bash
git add src/openjarvis/cli/chat_cmd.py tests/cli/test_chat_cmd.py
git commit -m "feat(chat): use generic grounded search flow"
```

### Task 5: Remove itinerary-only code and prove open-ended behavior

**Files:**
- Delete: `src/openjarvis/cli/_auto_search.py`
- Delete: `src/openjarvis/cli/_grounded_answer.py`
- Delete: `tests/cli/test_auto_search.py`
- Delete: `tests/cli/test_grounded_answer.py`
- Create: `tests/cli/test_general_life_information.py`

**Interfaces:**
- Consumes: the generic planner, preflight, grounded response, and chat command boundary.
- Produces: regression coverage showing no per-domain handler is required.

- [ ] **Step 1: Write table-driven open-ended behavior tests**

```python
@pytest.mark.parametrize(
    ("request", "style", "forbidden"),
    [
        ("과천시 오늘 날씨를 검색해 줘", "direct", ("방문 전 확인", "일정으로 정리")),
        ("집에서 하는 허리 운동법을 찾아줘", "steps", ("운영시간", "일정으로 정리")),
        ("무설탕 음료 세 개를 비교해 줘", "comparison", ("방문 전 확인",)),
        ("오늘 기술 뉴스를 찾아줘", "list", ("방문 전 확인",)),
        ("대전 하루 계획을 만들어 줘", "steps", ("날씨를 확인했어",)),
    ],
)
def test_unrelated_life_topics_share_one_generic_flow(request, style, forbidden):
    result = run_general_flow(request, style)
    assert result.style == style
    assert all(text not in result.rendered for text in forbidden)
```

Add cases for malformed plan repaired once, malformed answer repaired once, all searches failing, partial search success, other-city evidence rejected, private context absent from tool inputs, and unsupported action never claiming completion.

- [ ] **Step 2: Run the regression file before deleting old modules**

Run: `.venv/bin/pytest -q -p no:cacheprovider tests/cli/test_general_life_information.py`

Expected: all tests pass using the new generic modules.

- [ ] **Step 3: Remove itinerary modules and migrate references**

Use `rg` to find `_auto_search`, `_grounded_answer`, `AutoSearchResult`, `build_itinerary_prompt`, and `ItineraryItem`. Replace legitimate callers with generic interfaces and delete the four listed itinerary-only files. Leave no production compatibility alias that could accidentally restore itinerary routing.

- [ ] **Step 4: Run related CLI, web-tool, and memory tests**

```bash
.venv/bin/pytest -q -p no:cacheprovider \
  tests/cli/test_request_plan.py \
  tests/cli/test_assistant_preflight.py \
  tests/cli/test_search_evidence.py \
  tests/cli/test_grounded_response.py \
  tests/cli/test_general_life_information.py \
  tests/cli/test_chat_cmd.py \
  tests/tools/test_web_search.py \
  tests/memory/test_candidate_extractor.py \
  tests/memory/test_context_composer.py \
  tests/memory/test_personal_memory_service.py \
  -k 'not test_execute_no_api_key and not test_execute_tavily_error and not test_execute_import_error'
```

Expected: all selected tests pass with three network tests deselected.

- [ ] **Step 5: Run the three real network fallback tests with permission**

```bash
.venv/bin/pytest -q -p no:cacheprovider tests/tools/test_web_search.py \
  -k 'test_execute_no_api_key or test_execute_tavily_error or test_execute_import_error'
```

Expected: three tests pass.

- [ ] **Step 6: Commit removal and regression coverage**

```bash
git add -A src/openjarvis/cli tests/cli
git commit -m "refactor(search): remove itinerary-only response path"
```

### Task 6: Live acceptance, complete verification, and delivery

**Files:**
- Modify only if a live failure is first reproduced by a new failing automated test.

**Interfaces:**
- Consumes: `/Users/cksdndi95/.openjarvis/.venv/bin/jarvis chat` using feature-worktree source.
- Produces: verified runtime behavior and a pushed feature branch.

- [ ] **Step 1: Run live ordinary-chat acceptance**

Enter `안녕?` and verify normal conversation with no search source list and no forbidden old-topic resurfacing.

- [ ] **Step 2: Run live missing-location acceptance**

Enter `오늘의 날씨 검색해 볼래?` and verify it asks for a region instead of using English default queries or rendering an itinerary.

- [ ] **Step 3: Run live follow-up weather acceptance**

Reply `경기도 과천시야. 지금 시간은 오후 8시 24분이야.` and verify recent context is used, only public terms are searched, and a direct weather answer with relevant sources is rendered.

- [ ] **Step 4: Run live open-ended acceptance**

In fresh sessions test `집에서 할 수 있는 허리 운동법을 검색해 줘`, `무설탕 음료 세 개를 찾아 비교해 줘`, and the existing Daejeon itinerary request. Verify each uses the requested shape and does not inherit another topic's wording.

- [ ] **Step 5: Add a failing test before any live correction**

For a live failure, capture the exact plan document, evidence, answer document, and output. Add the smallest automated test reproducing it, observe the failure, then make one production change and rerun the focused test.

- [ ] **Step 6: Run Ruff and whitespace verification**

```bash
.venv/bin/ruff check --no-cache src/openjarvis/cli tests/cli
git diff --check
```

Expected: Ruff prints `All checks passed!` and `git diff --check` has no output.

- [ ] **Step 7: Run the complete repository suite**

Disable only pytest dead-symlink cleanup and force process exit after pytest returns, preserving all collection and test bodies:

```bash
.venv/bin/python -c 'import os,sys; import _pytest.tmpdir as tmpdir; tmpdir.cleanup_dead_symlinks=lambda path: None; import pytest; code=pytest.main(["-q","-p","no:cacheprovider","--basetemp=/tmp/openjarvis-general-agent-final","-k","not test_execute_no_api_key and not test_execute_tavily_error and not test_execute_import_error"]); sys.stdout.flush(); sys.stderr.flush(); os._exit(int(code))'
```

Expected: exit code zero, no failures, and only the three separately verified network tests deselected.

- [ ] **Step 8: Commit live-test corrections if needed**

```bash
git add src/openjarvis/cli tests/cli
git commit -m "fix(chat): finalize generic life information flow"
```

Skip this commit only when live acceptance needs no changes and the worktree is already clean.

- [ ] **Step 9: Push and verify the remote branch**

```bash
git push fork feature/developmental-personal-model
git rev-parse HEAD
git ls-remote fork refs/heads/feature/developmental-personal-model
git status --short
```

Expected: local and remote hashes match and `git status --short` has no output.
