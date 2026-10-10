# User-corrected two-player Bluff rules

Status: accepted rules are implemented in the corrected playable prototype.
The prototype exposes Random, Honest, Math, and an adaptive Flagship. User
gameplay approval was received on 2026-10-10: the user confirmed the intended
rules and baseline behaviors during local play. Bounded strength experiments
may now proceed; approval is not strength evidence. Do not
call the old runtime or weights correct for this game. No checkpoint or
historical result was changed by this note.

## Confirmed requirements

- Two players; retain the standard 52-card deck, 14 cards/player, and 1–4
  cards/play unless the user corrects those existing parameters.
- A round starter chooses any rank. No Aces opening restriction.
- That rank remains fixed for every play in the round. Actual played cards
  may differ from the claim (bluff). A mixed packet is false if ANY card
  differs. Three Kings plus a Two claimed as four Kings is a bluff.
- After a claim, the opponent may play at that same rank, pass, or challenge
  the latest packet. There is no obligatory separate call/pass response
  before choosing to continue the round by playing.
- Passing ends the round: recycle the center pile face down into the draw
  supply, shuffle, then draw one card. The other player starts the next round
  and can choose a new rank. The starter cannot pass before making a claim.
- The center pile is cleared on a pass and added back to the shuffled draw
  supply. The user called this supply the discard pile; implementation should
  distinguish the face-down center pile from the drawable supply explicitly.
- If the draw supply is empty, passing is unavailable. Continuing with an
  honest play/bluff remains possible; a challenge is also possible when there
  is a latest opponent packet. Empty supply must not force a challenge.
- A challenge inspects only the latest packet (one to four cards), not all
  previous packets. A false packet makes its player take the whole center
  pile; a true packet makes the challenger take the whole center pile.
- The challenge winner starts the next round with a new rank. Challenge
  pile cards go to the loser's hand, not also into the draw supply.
- An emptying play is automatically challenged: a voluntary decline is weakly
  dominated in two-player win/loss play. A caught final bluff returns the entire
  pile to the bluffer and the challenger starts a new round. A truthful final
  packet wins. Inspect/reveal only that final packet, not prior packets.
- Adding a new packet closes the previous challenge opportunity. Passing or
  resolving a challenge closes the old round. No retrospective challenges.
- There is no gameplay turn limit. A computational safety cutoff is recorded
  as truncation in experiments, not silently relabelled a draw or loss.

## Implementation boundary

Rules discussion and core prototype implementation are complete. The prototype
has been checked with engine, policy, server, and representative browser
sequences; these correctness checks do not establish strength. User gameplay
approval was received on 2026-10-10, clearing the strength-experiment gate.

## Worked examples exercised in the prototype

- A claims two Sevens. B can play one to four cards claiming Sevens without
  first drawing or challenging. A then has the same three response choices.
- A claims Sevens; B passes. Under the recycling proposal, the center pile is
  recycled/shuffled FIRST; B draws one; A starts a new round and may claim Kings.
- A lies claiming Sevens; B correctly challenges. A collects the entire pile;
  B starts a new round and may choose any rank.
- A truthfully claims Sevens; B incorrectly challenges. B collects the entire
  pile; A starts a new round and may choose any rank.
- A plays its last card(s); automatic challenge resolves it. If caught, A
  collects the pile and B starts a new round; otherwise A wins.

## Evidence and migration boundary

All prior complete-game evaluations/training trajectories are from a different
ruleset. Keep their source/config/hash records, but exclude them from intended-
game performance claims. The corrected prototype covers the core rules,
observations, frontend actions, and baseline policies. Historical runtime,
training, simulator, search, and deployment paths remain unmigrated. Consented
guest cross-game persistence is locally verified. Corrected-rule synthetic
diagnostics are underway (see AGENT_HANDOFF); human strength validation and
hosted release are still pending.

The prototype and baseline policies are implemented and user-approved for
gameplay. Prepare bounded paired/seat-swapped evaluations
and report truncations and uncertainty. Model reuse is a transfer hypothesis,
not validation. LLMs are experimental opponents or synthetic-data generators,
not the Flagship architecture or human validation.

## Current prototype roster and flagship direction

The corrected playable prototype currently offers one adaptive Flagship plus
three controls: Random, Honest, and Math. These are prototype policies, not
validated strength claims.

- Random: uniform legal action-category choice, then uniform quantity and
  card subset conditional on quantity; uniform rank at round starts only.
  This is hierarchical randomness, not uniform over every physical action.
- Honest: never voluntarily lies; plays its largest truthful legal packet,
  passes when unable to continue honestly, and voluntarily challenges only a
  certified contradiction. At round starts choose a largest held rank group.
- Math: card-conserving beliefs, fixed opponent prior, and transparent short-
  horizon action values. No identity-specific behavioral learning or long search.
- Adaptive Flagship: the same short-horizon backbone plus smoothed contextual
  call/revealed-bluff evidence. Guest evidence lasts within room rematches;
  with Data opt-in and an available database, it checkpoints on completion or
  interruption and restores for that returning browser in a new room.
  Test adaptation without confounding it with a different planner. Unknown
  labels stay unknown.
- A non-adaptive search control remains a research candidate, not a current
  player-menu option or implemented control. It could help isolate search and
  persistent-memory contributions in later experiments.

Flagship proposal: certified card constraints plus coherent hidden-card beliefs,
contextual opponent forecasts, bounded imperfect-information lookahead, mixed
actions when appropriate, and conservative handling of sparse/stale evidence.
Within-game card-location beliefs reset at each deal; behavioral identity memory
is separate, opt-in and uncertainty-aware. Wrong challenges can reveal useful
information but transfer known cards to the caller; model current locations,
not simply "opponent has cards they once revealed". Search continuations must
use each actor's legal observations, not leaked sampled hidden states.

Neural detector/value/policy models are candidate approximators only after a
correct engine and competitive simple predictors/planners exist. Correct-rule
self-play can provide synthetic training data but does not establish human
transfer. PureNN and AcademicBeast remain historical, not validated new-rule
baselines. No automatic reuse/promotion of their checkpoints.

## Recycling reasoning and source boundary

For an ordinary pass with `P >= 1` center cards and `D` drawable cards,
recycle-then-draw yields `D' = D + P - 1 >= D`. Plays and challenges do not
deplete the draw supply. Thus with initial supply 24 and these exact rules,
an empty draw supply is unreachable. An empty-supply guard is defensive for
custom fixtures/variants, not a normal endgame mechanic. Recycling does not
guarantee termination: policies can cycle; evaluator budgets must not quietly
become game rules or censor long games without reporting it.

Recycling means prior packets cannot be labelled permanently removed. It does
NOT invalidate current-hand disproof: own copies `k` plus latest claimed
quantity `q > 4 - k` is impossible. Known own packet identities still in the
center also constrain ownership within the round. After recycling/transfers,
tracking must update location uncertainty, not treat historical counts as
cards gone forever. Even with permanent face-down removal, an opponent's
unrevealed claims are not verified discarded rank identities.

Rules reference: [Pagat's fixed-rank I Doubt It/Bluff](https://www.pagat.com/beating/doubt.html)
describes fixed-rank rounds, winner-led challenges, and hidden removal of a
passed-out stack. It is not identical to our two-player draw-on-pass variant;
do not claim recycling is a universal traditional rule. The desired product
retains the fixed-rank deception core, with an explicit draw/recycling house rule
if approved. Do not add unrelated force-challenge mechanics.

Relevant primary research: [Bitan & Kraus](https://arxiv.org/html/1709.09451v2)
integrates human-claim prediction with search in a different Cheat variant;
[Bayes' Bluff](https://arxiv.org/abs/1207.1411) studies opponent-strategy posteriors;
[Data Biased Robust Counter Strategies](https://proceedings.mlr.press/v5/johanson09a.html)
addresses exploiting observations while protecting against misleading or changed
opponents. None validates our previous rules or dictates the best draw rule.
