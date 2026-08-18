# Model-led Harness Design

## Status

Proposed for user review.

## Problem

The current chat path asks several components to reinterpret the same user
request. A local planner classifies the request, builds a strict search plan,
and may repair that plan. Search results are filtered against planner-generated
terms. A second model must then emit a strict citation format, and another pass
may repair that response. Personal memory follows a similar pattern: a
deterministic language parser and a background extractor independently
reinterpret a completed exchange, while static profile files can remain in
conflict with newer stored claims.

These layers can reject or distort a meaning the conversation model already
understood. The observed failures are harness failures rather than evidence
that the local model cannot handle ordinary conversation:

- The configured model selected `web_search` and generated a useful public
  query for a Korean weather request without the planner.
- The model understood an assistant-name correction in the live conversation,
  but the later single-exchange memory pass lost its context.
- The search tool returned link-only snippets without the requested weather
  value, after which the model guessed a value. Tool-result quality therefore
  matters alongside model autonomy.

## Governing Principle

The model owns semantic judgment. The harness supplies the information and
capabilities needed for that judgment, then executes, validates, classifies,
and persists the result.

The harness must not silently reinterpret a user request or replace the
model's semantic decision with a weaker heuristic. It may block or request
confirmation only at explicit machine-checkable boundaries such as privacy,
permissions, destructive effects, unsupported tool arguments, missing source
content, or invalid persistence structure.

## Responsibility Boundary

### Model responsibilities

- Understand the user's intent from the current message and recent dialogue.
- Decide whether to answer directly, ask a question, or use a tool.
- Select tools and form tool arguments.
- Decide whether returned information is sufficient and, when needed, use
  another read-only tool or state that the answer is unavailable.
- Interpret corrections, preferences, constraints, facts, episodes, and
  relationships in conversation.
- Produce a natural final response.

### Harness responsibilities

- Assemble recent dialogue, canonical personal memory, tool descriptions, and
  tool results without changing their meaning.
- Enforce privacy, capability, permission, and approval boundaries.
- Execute tool calls and return structured success, error, source, and content
  data to the same model.
- Bound turns, time, context size, and retries.
- Validate storage structure and verify that personal-memory evidence is an
  exact user-authored excerpt.
- Classify and persist model proposals without performing a second semantic
  interpretation.
- Record observable reasons for blocked calls, rejected structures, and
  retries.

### Shared boundary

For low-risk read-only work, the harness should be permissive and return tool
errors to the model for recovery. For external writes, financial actions,
messages to other people, sensitive disclosure, or destructive operations,
the model proposes an action and the harness obtains approval or blocks it.

## Conversation and Tool Flow

`jarvis chat` will use one tool-capable conversation agent as the semantic
owner of a turn.

```text
user message
  -> context assembly
  -> conversation model
       -> natural answer, or
       -> tool call
            -> privacy / permission / argument checks
            -> tool execution
            -> structured tool result
            -> same conversation model
  -> final natural answer
  -> durable exchange archive
```

The generic request planner, requirement-term gate, strict grounded-response
JSON contract, and response-repair pass will be removed from the ordinary chat
critical path. They must not be replaced by new domain-specific planners.

### Search tools

Search remains a general capability rather than a collection of domain
models. The model chooses `web_search` itself. Search results must expose:

- query and retrieval time;
- result title, URL, and snippet;
- whether full page content was fetched;
- fetched page content when available;
- an explicit error when content cannot be obtained.

A general read-only page-fetch capability will let the model follow a search
result when snippets do not contain the requested value. Exact values must
come from tool content. The model is instructed not to invent missing values,
but the harness does not force the final prose through a brittle citation JSON
format. Stronger verification remains available as an opt-in policy for
medical, legal, financial, or other high-stakes answers.

## Personal Memory Flow

The raw exchange remains the durable source record. Memory processing receives
a bounded recent-dialogue window, the completed user and assistant turn, and
relevant canonical memory. It proposes typed changes such as create,
reinforce, refine, supersede, or hold-conflict.

```text
completed exchange plus recent dialogue
  -> local semantic memory proposer
  -> typed proposal with exact user evidence excerpts
  -> structural and provenance validator
  -> canonical personal-memory store
  -> response-time context projection
```

The validator checks that required fields are present, the operation is
allowed, and every evidence excerpt exists verbatim in a user message. It does
not use regular expressions to decide what the user meant and does not discard
direct corrections merely because their kind is a direct rule.

Direct user corrections and explicit constraints are eligible for immediate
supersession after provenance validation. Inferred patterns, motives, and
personality claims continue through recurrence, conflict, reflection, and user
confirmation gates.

The personal-memory database becomes the canonical source for changing user
and assistant facts. `USER.md`, `MEMORY.md`, and legacy fact files may supply
initial static material, but they must not remain independent competing
authorities. Dynamic prompt context is projected from the canonical store.
Pending work is durable, and the bounded recent dialogue remains available
across restart so that an immediate shutdown cannot erase the context needed
to understand a short correction.

## Failure Behavior

- A malformed model tool call is returned to the same model with the exact
  schema error and at most one bounded retry.
- A read-only tool failure does not become a generic misunderstanding message;
  the model receives the failure and can retry, ask a useful question, or
  explain the limitation.
- The model must not treat search results that lack the requested value as
  factual evidence for an exact value.
- A malformed memory proposal leaves the raw exchange intact for later retry.
- Conflicting direct memories are shown to the model as an explicit conflict;
  the latest user correction may supersede older claims when its evidence is
  clear.
- No component silently substitutes a fallback semantic interpretation.

## Migration

1. Preserve all archived exchanges and accepted claim provenance.
2. Stop injecting legacy dynamic facts and stale profile values as parallel
   authorities.
3. Import still-valid static profile entries into the canonical store with
   legacy provenance.
4. Mark contradictory older claims superseded only when a newer direct user
   correction provides evidence; otherwise retain the conflict for user
   review.
5. Keep user controls for listing, correcting, suppressing, exporting, and
   deleting personal memory.

## Evaluation

Evaluate the same local model under the old and new harnesses. The test corpus
must include natural paraphrases rather than only implementation-shaped
phrases.

Required end-to-end cases:

- ordinary chat does not invoke a tool;
- explicit and implicit current-information requests invoke search;
- a Korean weather request produces a useful query and does not invent a
  missing value;
- search failure results in recovery or an honest limitation;
- a correction such as `공박사라고.` is understood using recent dialogue;
- a corrected identity survives process restart and supersedes stale profile
  data;
- direct prohibitions survive restart and are not reintroduced by legacy
  memory;
- ambiguous memory is retained as evidence but not promoted as fact;
- private context is not copied into a public search query;
- external writes still require approval.

Primary measures are task success, correct tool selection, unsupported factual
claims, restart recall, stale-memory conflicts, latency, model calls per turn,
and tokens per successful task. Traces must identify whether a failure came
from model judgment, missing context, poor tool output, blocked permission, or
invalid persistence structure.

## Non-goals

- Building a separate model for weather, shopping, exercise, or each future
  life domain.
- Removing permission, privacy, provenance, or destructive-action safeguards.
- Letting web content become instructions.
- Treating every conversational remark as permanent memory.
- Requiring a multi-agent system for ordinary chat and search.

## Acceptance Criteria

1. One conversation model owns semantic decisions for an ordinary chat turn.
2. The model can call enabled read-only tools directly from `jarvis chat`.
3. No planner or citation-repair model can replace a valid low-risk answer with
   a generic failure message.
4. Exact public facts in an answer are traceable to actual tool content.
5. Memory proposals are made with recent dialogue and exact user evidence.
6. Direct corrections can supersede stale claims without a language-specific
   meaning parser.
7. The canonical personal-memory store is the only dynamic source of truth
   injected into a new session.
8. Privacy and high-risk action boundaries remain deterministic, explicit, and
   auditable.
