# Personal Memory Harness

## Status

Implemented first slice: local conversation archive and asynchronous memory
candidates. Pattern memory and the core user model are explicitly deferred.

## Goal

Extend OpenJarvis without replacing its existing document-retrieval store or
automatic JSONL fact memory. Every completed personal conversation must first
be retained as local, auditable source material. A local model may then derive
provisional memory candidates from that source material without affecting reply
latency or injecting unverified claims into future prompts.

## Decisions

- Personal conversation and memory data stay on the local machine.
- Memory processing must use a local engine. An unavailable or non-local
  engine prevents candidate generation; it must never fall back to a cloud
  provider.
- The existing `memory_facts.jsonl` store and document RAG backends remain
  compatible and operational.
- A memory candidate is not a fact. It is unavailable to prompt injection
  until a later evaluation/promotion phase accepts it.
- The initial archive is a separate SQLite database at
  `~/.openjarvis/personal_memory.db`, rather than a new table in the document
  retrieval database.

## Scope

### Included

1. A durable local archive for completed exchanges.
2. An idempotent, asynchronous candidate-generation queue backed by the
   archive.
3. Candidate provenance, state, importance, and extraction metadata.
4. Integration hooks for both standard chat and managed Companion agent turns.
5. Local-engine enforcement and observable failure states.
6. Unit and integration tests for durability, deduplication, queue pressure,
   and local-only processing.

### Deferred

- Automatic promotion of candidates into Fact, Episode, Pattern, or Core User
  Model memory.
- Reflection scheduling and cross-exchange pattern discovery.
- Candidate retrieval and prompt injection.
- Migration of the existing JSONL fact store.
- Redesign of the experimental behavior-analysis tools.

## Architecture

The write path must make the raw exchange durable before any lossy or
best-effort operation occurs:

```text
completed chat or Companion turn
  -> record_and_publish_completed_exchange()
  -> PersonalMemoryArchive (SQLite transaction)
  -> CHAT_EXCHANGE_COMPLETED event containing exchange_id
  -> PersonalMemoryService background worker
  -> memory_candidates row (pending)
```

`record_and_publish_completed_exchange()` is the shared boundary used by server routes,
CLI chat, and the managed-agent executor. It accepts user text, assistant text,
source, and optional agent/session identifiers. It first writes the exchange
and only then publishes the event. Replaying a call with the same exchange ID
is a no-op.

`PersonalMemoryService` subscribes to the completed-exchange event. Its queue
is only a scheduling mechanism: the archive is the source of truth. A queue
overflow or worker restart leaves the exchange archived and eligible for a
future retry; it cannot discard the conversation itself.

## Local Data Model

### `conversation_exchanges`

- `id`: stable exchange UUID, primary key.
- `created_at`: UTC timestamp.
- `source`: `cli.chat`, `server.chat`, or managed-agent source.
- `agent_id` and `session_id`: optional origin identifiers.
- `user_text` and `assistant_text`: original exchange contents.
- `content_hash`: deterministic hash for diagnostics and duplicate detection.
- `archived_at`: timestamp of successful local persistence.
- `candidate_state`, `candidate_attempts`, `candidate_error`, and
  `candidate_claimed_at`: durable worker state (`pending`, `processing`,
  `failed`, or `complete`), separate from candidate-review status.

### `memory_candidates`

- `id`: candidate UUID, primary key.
- `exchange_id`: foreign key to the supporting exchange.
- `kind`: initially `fact` or `episode`.
- `content`: structured candidate text.
- `importance` and `confidence`: bounded numeric estimates.
- `status`: `pending`, `accepted`, `rejected`, or `superseded`.
- `engine_id` and `extractor_version`: local provenance.
- `created_at` and `updated_at`.

An exchange can have zero or more candidates. Candidates always point back to
the exact conversation that supports them.

## Candidate Processing

The worker loads an archived exchange by ID and calls a deterministic,
local-only extractor prompt. It accepts only the configured loopback host of a
known local engine (or an existing local `gemma_cpp` model path); cloud,
multi-engine, and non-loopback configurations fail closed. It validates the
structured response, stores each candidate transactionally, and records the job
outcome. Errors use the exchange ID rather than conversation text and leave the
exchange in an eligible-for-retry state. Candidate generation must never block
or fail a user-facing response.

The first slice does not claim that a candidate is true. It only records what
the local extractor proposes and the evidence from which it was derived.

## Integration Boundaries

OpenJarvis already publishes `CHAT_EXCHANGE_COMPLETED` for standard CLI and
server chat, and `MemoryService` uses it for best-effort Fact extraction. The
new shared record boundary will preserve that compatibility while enriching the
event payload with an `exchange_id` and optional origin metadata.

Managed agents persist messages in the agent manager. On successful
finalization, a pending user message and its response now pass through the same
completion boundary once. Scheduled ticks without a pending user message create
no personal exchange. This does not inject candidates or alter existing
Fact/RAG prompt behavior in this slice.

The current working-tree Companion additions are intentionally not copied into
this feature branch. Their integration point is specified here so it can be
applied when those independent changes are committed or merged.

## Privacy and Operations

- Database paths are under the OpenJarvis local configuration directory.
- Archive and candidate generation reject non-local engine configuration.
- No network synchronization, cloud fallback, or external telemetry is added.
- Future management commands must support listing and deleting personal-memory
  records, but destructive operations are outside this first slice.
- Logs must use exchange IDs and status, not conversation body text.

## Acceptance Criteria

1. Each completed standard chat exchange is stored once locally before
   candidate work begins.
2. A duplicate exchange ID never creates a second archive row or duplicate
   candidates.
3. A full worker queue, malformed extractor output, worker exception, or
   restart cannot remove an already archived exchange.
4. Candidate processing refuses non-local engines and makes no cloud fallback.
5. Existing automatic Fact extraction, Fact listing, document memory retrieval,
   CLI chat, and server chat continue to work.
6. Managed-agent integration is generic and records only completed turns that
   began with a pending user message; unrelated Companion files remain outside
   this feature branch.

## Follow-up Sequence

After this slice is stable, add evaluator-driven promotion, then scheduled
reflection over accepted evidence, then Pattern Memory and an explainable Core
User Model. Each stage must add a test corpus and explicit promotion criteria
before its data is injected into prompts.
