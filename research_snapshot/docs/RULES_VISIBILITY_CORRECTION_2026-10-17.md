# User correction: challenge visibility, 2026-10-17

## Accepted visibility rule

After **any challenge**, both players see the complete center pile in its
original play order. This includes a voluntary challenge and the automatic
challenge of a final, emptying play. The challenge outcome is still determined
only by the latest packet: if that packet is false, its player takes the whole
pile; if it is true, the challenger takes the whole pile. Earlier packets do
not affect the result, even though their identities are now public.

Passing is different. The pile is recycled face down into the shuffled draw
supply, so its card identities remain hidden. The existing fixed-rank,
play/pass/challenge, round-ending and pile-transfer rules remain in force.

## Information and evidence boundaries

Public card visibility must be represented independently from `bluff_called`.
That field describes whether a particular packet was challenged; it must not
become a shorthand for “these card identities are public.” A separate,
explicit public identity-reveal marker is needed for each packet exposed by a
challenge. Truth labels can be derived for every exposed packet. Keep the
challenge result and caller correctness attached to the latest packet only.
Do not attribute an earlier packet's
public exposure to a challenge against that packet.

Before a challenge, opponent packet identities and truth remain hidden. Own
cards remain private to their owner, and a pass must not expose the recycled
cards. A resolved challenge may expose public identities without making every
packet a separately challenged action. Preserve consent, deletion, and
research-log controls; privileged research transcripts remain distinct from
player-visible observations.

## Implementation status and migration boundary

Local implementation now covers the engine, shared action redaction, public
packet truth labels, card-location bounds, adaptive frequency evidence,
collector-channel compatibility, search keys, API reveal payloads, browser log
handling, and Rules modal copy. A distinct `cards_revealed` action flag is set
on all packets, while only the latest packet has `bluff_called=True`. The
ruleset/event storage label is `bluff-public-pile-v3`, transcript provenance is
`guest-events-v2`, and profiles use a fresh `round_adaptive_v3` namespace. Old
profiles are retained, not silently converted. Owner metadata inventory can
display both versions separately.

V3 owner timeline export/projection/viewer and transcript/transition integrity
now support homogeneous, correctly paired v2/v3 labels. V3 disclosure validation
matches the complete ordered pile to the preceding privileged play ledger and
rejects forged/mistimed disclosures. Actual server-emitted synthetic events
cover voluntary and automatic calls, hidden pass recycling, and historical
projection compatibility; these are correctness checks, not strength results.

Bounded health and baseline audit tools now use a shared quality gate and
display v2/v3 separately; see COLLECTION_QUALITY_V3_MIGRATION_2026-10-17.md.

This is **not deployed or a completed migration**. Specialized research
adapters and historical replay tests still require review.

The LLM console/series and dual-policy wrapper now forward full-pile legal
observations. Delayed opening/response learners consume validated public labels
even when their viewer did not collect the pile, with deduplication and
prevalidation before response-model mutation. Use the new v3 win-only tester
prompt; older prompts/artifacts remain historical. These adapter checks and
scripted replay tests are not newly played LLM games or strength evidence.
A broad test run exposed compatibility failures;
focused passing checks do not establish whole-project readiness. No strength
run or policy promotion is authorized by these code changes. Historical
artifacts remain unchanged and cannot be replayed under the new engine as if
the rules were identical.

Keep older ruleset identifiers, source hashes, checkpoints, logs, and results
as historical evidence. Version the visibility semantics in ruleset and
artifact provenance before using newly generated observations or results;
do not mix them silently with data produced under the previous reveal rule.

## Bounded integration checklist

- Update engine challenge resolution to make the complete ordered pile
  available to both players after voluntary and automatic final challenges;
  preserve latest-packet-only result and transfer logic.
- Update public action projection, API/browser payloads, and UI so each
  exposed packet has a distinct public-identity marker; retain redaction before
  resolution and through face-down pass recycling.
- Keep privileged audit logging and consent/deletion behavior intact. Record
  the new visibility provenance without exposing private hands or unrelated
  hidden state.
- Audit card memory, pickup inference, and behavioral learning for double
  counting: distinguish known identities, inferred ownership after pile
  transfer, challenge-response evidence, and truth labels.
- Update information-state/search keys and simulated actor views to include
  only identities public to that actor at that point in the game.
- Version ruleset/artifact provenance and keep old evidence on its original
  visibility semantics.
- Add focused tests for ordered pile exposure after both challenge types,
  latest-packet-only resolution, pre-challenge redaction, hidden pass/recycle,
  actor-specific search observations, and learning/provenance deduplication.

This checklist is scoped to visibility migration; passing it would establish
correctness for these cases, not policy strength or human benefit.
