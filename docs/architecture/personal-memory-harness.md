# Personal Memory Harness

## Status and purpose

The personal-memory harness is a local archive for changing user and assistant
context. Its default path is `~/.openjarvis/personal_memory.db`. It complements
document retrieval and the legacy JSONL Fact Memory; it does not replace either
storage primitive. After canonical-profile activation, it is the changing
profile authority for `jarvis chat` when response-time memory context is enabled.

The model owns semantic interpretation. The harness supplies bounded dialogue,
requires a typed proposal, verifies exact user evidence, applies deterministic
lifecycle rules, and projects only accepted state. It does not use
language-specific phrase rules to decide what the user meant.

## Responsibility boundary

The local model decides:

- whether the current user message contains durable personal information;
- the candidate kind, stable dotted subject, temporal scope, and correction
  target;
- whether a new observation supports, contradicts, or qualifies an existing
  schema when reflection is enabled.

The harness decides only enforceable boundaries:

- the response has the required schema and bounded sizes;
- confidence and importance are finite numbers from `0.0` through `1.0`;
- every `evidence_excerpt` is an exact substring of the current user message;
- assistant text and earlier dialogue cannot become evidence;
- only allowed state transitions are persisted;
- optional external reflection never automatically submits archive or profile
  bodies, and known sensitive patterns must pass its deidentification gate.

This split lets a short correction use conversational meaning without turning
the harness into an ever-growing intent router.

## Durable lifecycle

```text
completed user/assistant exchange
  -> archive the raw exchange in local SQLite
  -> publish CHAT_EXCHANGE_COMPLETED
  -> enqueue durable work by exchange ID
  -> local model proposes schema-constrained candidates
  -> validate exact current-user evidence
  -> evaluate and apply an auditable state transition
  -> project bounded active memory into a later response
```

`record_and_publish_completed_exchange()` is shared by CLI, server, and
managed-agent completion paths. Archiving happens before the completion event;
replaying the same exchange ID is idempotent. The personal-memory worker queues
durable IDs rather than raw conversation text. The common completion event
still carries the existing compatibility payload used by other memory
consumers.

SQLite is the source of truth. The in-memory queue is only a wake-up mechanism.
`memory_jobs` retains idempotency keys, priority, attempts, retry time, and
state. Startup returns interrupted work to the queue, requeues zero-candidate
results created by an older extractor version, and schedules unevaluated
candidates. A queue overflow, shutdown, invalid model response, or engine error
therefore cannot erase the archived exchange.

## Proposal and evidence contract

Extractor version 8 receives the current completed exchange and at most six
earlier exchanges in chronological order, scoped to the same session when a
session ID exists and otherwise to the same source. The earlier user/assistant
turns provide meaning only. The current user message is the sole evidence
source.

Each material proposal-prompt revision advances this provenance version.
Startup may therefore reopen a completed zero-candidate exchange from an older
version through the ordinary durable job lifecycle; a successful version-8 run
then prevents the same exchange/version pair from being scheduled again.

The model may propose up to five atomic candidates from this closed set:

- `fact`
- `episode`
- `preference`
- `constraint`
- `correction`
- `role_preference`
- `capability_boundary`
- `hypothesis`

Each candidate includes content, importance, confidence, temporal scope, a
stable subject, an optional target claim, and an exact evidence excerpt.
The proposal prompt directs the model to return no personal-memory candidate for
a request that only seeks current external-world information, and to return an
empty array for greetings or messages without useful personal evidence. The
validator continues to enforce structure and exact evidence rather than
reclassifying meaning itself.

Malformed output is never partially salvaged. It records a bounded error code
and remains retryable; the harness does not invent a semantic fallback.

## Evaluation, correction, and supersession

A candidate is provisional until `MemoryEvaluator` rechecks the supporting
exchange and evidence excerpt. Accepted atomic claims link to immutable evidence
and an auditable decision.

Direct user rules (`constraint`, `correction`, `role_preference`, and
`capability_boundary`) can take effect immediately after provenance validation.
A correction supersedes an explicitly targeted active claim. If no stored claim
ID was available to the model, a correction with a non-generic stable subject
supersedes active direct claims with that same subject. The old claim and its
evidence remain stored in `superseded` state; they are not rewritten.

Delayed background evaluation cannot reverse a newer direct statement for the
same subject. The archive orders direct evidence by source-exchange time and
exchange ID, then by candidate/evidence time and ID for a deterministic tie.
Inside the same write transaction, an older direct rule is rejected when a
newer same-subject atomic claim is active. The same chronology rule applies to
direct-source facts and preferences, so delayed model classification cannot
make an older observation coexist with a newer role preference or correction.
A genuinely newer explicit correction still applies normally. Stale attempts
and the active claim remain auditable without rewriting evidence.

Accepted non-direct facts and preferences first remain atomic claims. Schema
consolidation and reflection run only after recurrent cross-session evidence or
a conflict trigger. A model-proposed ambiguous or contradictory relation to an
existing schema is held as disequilibrium. Broad identity, health,
mental-health, personality, and relationship-motive interpretations require
user confirmation before a schema can activate.

## Response-time projection

`ContextComposer` builds a bounded view rather than copying the archive into
the prompt:

| Section | Selection | Default cap |
|---|---|---:|
| Direct user rules | Every active direct rule, newest first | 5 |
| Current state | Relevant non-direct claims with `temporal_scope=current` | 3 |
| Confirmed schemas | Relevant active schemas and their conditions | 8 |
| Episodes | Relevant non-current episode claims | 5 |
| Unresolved | A relevant pending clarification, when needed | 1 |

Current states, schemas, episodes, and pending clarification use a conservative
local relevance check. Direct rules do not depend on query keywords. Raw
evidence is not rendered into this system context.

Separately, `jarvis chat` carries at most six incomplete user messages across a
restart as chronological `user`-role dialogue. They are not system rules, are
not evidence for a new candidate, and never include assistant text. This
continuity window allows an immediate restart to preserve the context needed
for a short follow-up while unfinished extraction remains retryable.

This raw continuity window is scoped by the conjunction of archive `source`
and `session_id`; canonical accepted claims remain available globally. CLI
chat records use `source=cli.chat` and a stable session key. The default key is
`default`, so an ordinary CLI restart continues the same pending dialogue;
`jarvis chat --session-id <key>` selects an independent CLI conversation.
Server chat, managed-agent exchanges, and every other CLI session are excluded
even when their pending work lives in the same personal-memory database.

All personal context composition requires an allowlisted local response engine.
If `agent.context_from_memory=false`, collection may continue but personal
context is not injected.

## Modes and canonical authority

`personal_memory.mode` has three behaviors:

- `off`: no personal-memory service or response context;
- `shadow` before canonical activation: collect and evaluate, but expose only
  active `assistant_behavior` constraints, role preferences, and capability
  boundaries to responses;
- `active`: expose the full composed context only after the separate legacy
  rollout gate is release-ready.

After `canonical_profile_active=1`, canonical context is the dynamic authority
even when the configured mode remains `shadow`; the pre-cutover narrow shadow
filter no longer hides accepted canonical state. Canonical-profile activation
and the legacy JSONL rollout flag are independent controls.

For `jarvis chat`, canonical activation keeps `SOUL.md` but excludes `USER.md`,
`MEMORY.md`, and their named-persona equivalents from prompt construction.
Legacy JSONL facts are also suppressed in that CLI path, so the SQLite archive
is the only changing profile authority. The server prompt path has not yet been
cut over by this helper; operators must not assume activation changes every
entry point.

Before cutover, confirm `[personal_memory]` is enabled, its mode is not `off`,
and `agent.context_from_memory=true`. Activation still excludes static
`USER.md`/`MEMORY.md` when context injection is false; in that configuration
`jarvis chat` retains `SOUL.md` but receives no dynamic personal context.

## Reversible profile cutover

Run staging before activation:

```text
jarvis memory stage-canonical-profile
```

Staging does not activate anything and does not alter the source profile files.
For each configured regular `USER.md` or `MEMORY.md`, it creates a private
descriptor-based snapshot under
`<archive-directory>/.canonical-profile-backups/`, forces directories to
`0700` and files to `0600`, and records source path, backup path, and SHA-256 in
the archive manifest. Symlinks, non-regular files, invalid UTF-8, unsafe path
changes, and platforms without the required no-follow POSIX operations fail
closed. Non-empty bullets are imported only as low-trust `LEGACY_IMPORT`
candidates with `0.1` confidence and importance; they do not become active
automatically.

Before activating, confirm that:

1. the archive is a readable, non-symlink SQLite file and its integrity check is
   `ok`;
2. every manifest backup is a regular non-symlink file;
3. backup paths match the manifest in exact order;
4. each snapshot SHA-256 still matches;
5. direct corrections needed at cutover are active and conflicting claims are
   superseded.

Then run:

```text
jarvis memory activate-canonical-profile
```

With no options, the command uses the stored manifest. Repeated
`--backup-path` values are accepted only when they exactly match that manifest.
Successful activation records the manifest hash and an audit decision.

Activation performs the archive, ordered-path, file-type, hash, private-mode,
owner, and descriptor-identity checks internally. It records the validated
backup identities with the canonical marker. The staging command prints
aggregate counts, not manifest paths or hashes; independent inspection
currently requires trusted local maintenance tooling because there is no
dedicated manifest-inspection CLI.

To leave canonical mode without overwriting the profile files that currently
exist, run:

```text
jarvis memory deactivate-canonical-profile
```

The command repeats the exact ordered-manifest, SHA-256, `0700` directory,
`0600` file, owner, non-symlink, regular-file, and descriptor-identity checks.
Attestation schema version 2 covers device, inode, uid, mode, size, and
nanosecond mtime/ctime for the root, each snapshot directory, and every file.
An installation activated by an older schema must rerun the activation command
against its unchanged exact manifest and backups before using deactivation.

All root, snapshot-directory, and file descriptors remain pinned across the
cutover. Each validation pass hashes every pinned file first, freshly reopens
every absolute root/snapshot/file path without following symlinks, compares all
attested fields, and finally repeats a global full-field check of every pinned
descriptor. In the cutover transaction, metadata and the non-content audit row
are first staged as uncommitted changes; this validation then runs as the last
application check, and failure rolls the transaction back. Only a successful
check permits commit. Current `USER.md` and `MEMORY.md` bytes are never restored
or overwritten; those current sources simply become eligible for static
injection again. Mutations observed by the checks fail closed and leave both
canonical metadata and audit state unchanged.

POSIX filesystem state and SQLite do not share a kernel-level transaction.
There is therefore an unavoidable interval between the last filesystem
observation and SQLite commit, and multiple paths can only be observed
sequentially. This command narrows that boundary; it does not claim protection
against a same-user actor that can keep mutating private backups after the last
observation.

Preserve the archive, manifest, and snapshots after deactivation. This command
is a safe authority rollback, not an automatic file restore. Do not use
`personal-delete-all` as rollback: it intentionally retains archive metadata
and filesystem backups.

## Privacy boundary

- The archive and extractor remain local. Extraction accepts known local
  engines only when they use loopback addresses (or an existing local
  `gemma_cpp` model path); there is no cloud or remote-network fallback.
- In `jarvis chat`, a non-local response engine receives the user's live request
  as part of that response path, but it does not receive personal-memory context
  or static `USER.md`/`MEMORY.md`. Other prompt entry points are not covered by
  this CLI-specific guarantee.
- Optional external reflection requires an explicit analysis request, a
  deidentified general query, and mode-appropriate consent. Sensitive requests
  require consent even in `allowed` mode. External results are stored separately
  as external knowledge and never become personal evidence automatically. The
  deidentification check recognizes bounded sensitive patterns; it is a
  heuristic guard, not proof that arbitrary text is anonymous.
- Logs and background errors use IDs, counts, and bounded reason codes rather
  than conversation bodies.
- The archive is local storage, not an encryption guarantee. Snapshot
  permissions are enforced, but the SQLite archive is neither encrypted nor
  forced to mode `0600` by this feature; its file mode depends on the operating
  system and process umask.
- Personal-memory HTTP controls accept loopback clients only.
- `personal-list` and `personal-explain` intentionally print stored material;
  use them only in a private terminal.

## Inspection and user controls

```text
jarvis memory personal-list
jarvis memory personal-explain SUBJECT_ID
jarvis memory personal-suppress SUBJECT_ID [--reason TEXT]
jarvis memory personal-restore SUBJECT_ID
jarvis memory personal-correct SUBJECT_ID REPLACEMENT --user-text TEXT
jarvis memory personal-delete SUBJECT_ID [--include-raw-evidence]
jarvis memory personal-delete-all --confirm-token "DELETE ALL PERSONAL MEMORY"
```

`personal-correct` records the supplied user text as new user-confirmed
evidence. For a claim it creates a correction and supersedes the old claim
without rewriting history; for a schema it creates a refined schema version.
Suppression affects the next composition without deleting evidence.
Single-subject deletion retains raw evidence by default. Deletion with
`--include-raw-evidence` can be refused when that evidence is still linked to
another schema. Bulk deletion requires the exact confirmation token shown
above.

Legacy JSONL Fact Memory uses separate commands:

```text
jarvis memory personal-import-legacy [--path PATH] [--backup|--no-backup]
jarvis memory personal-rollout-status --backup-path PATH [--manual-override] [--activate]
```

That rollout gate is not the canonical profile cutover described above.

## Failure behavior

- The user-facing answer does not wait for candidate extraction.
- Invalid output, engine failure, queue pressure, and interrupted shutdown leave
  raw exchanges durable and work retryable.
- Direct corrections are never discarded merely because they are direct rules.
- Evidence absent from the current user message is rejected, even when it
  appears in assistant text or recent dialogue.
- External reflection that lacks request, consent, or deidentification remains
  local and performs no provider call.
- If an existing canonical archive is unreadable or corrupt, static dynamic
  files are not silently reintroduced as a competing authority. A missing
  archive is treated as inactive, so operators must preserve the archive during
  recovery.
