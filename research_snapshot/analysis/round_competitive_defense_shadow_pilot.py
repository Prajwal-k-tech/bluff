"""Paired same-claim defensive shadow on stable and switched synthetic series."""
import hashlib
import json
from pathlib import Path
import time

from analysis.round_defense_reset_shadow import DefenseResetShadow
from analysis.round_predecision_bluff_audit import _game_trace, _metric
from analysis.round_competitive_switch_actor import RoundCompetitiveSwitchActor


def main():
    seed_base = 2413001001
    path = Path('data/round_competitive_defense_shadow_pilot_20261029.json')
    if path.exists():
        raise FileExistsError(path)
    files = ('analysis/round_competitive_defense_shadow_pilot.py', 'analysis/round_defense_reset_shadow.py',
             'analysis/round_predecision_bluff_audit.py', 'analysis/round_competitive_switch_actor.py',
             'analysis/round_arena.py', 'bots/round_opening_public_expert_bot.py',
             'bots/round_public_expert_response_bot.py', 'bots/round_policy.py',
             'bots/round_observation.py', 'game_v2.py')
    hashes = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    rows = []
    start = time.monotonic()
    for identity in range(2):
        for regime in ('stable', 'switch'):
            profile = opponent_profile = None
            for gi in range(2):
                seed = seed_base + identity*10000 + gi
                focal = DefenseResetShadow(seed+10)
                if profile is not None:
                    focal.restore_opponent_profile(profile)
                before = hashlib.sha256(json.dumps(focal.to_dict(), sort_keys=True).encode()).hexdigest()
                phase = 3 if regime == 'switch' and gi == 1 else 0
                opponent = RoundCompetitiveSwitchActor(seed+20)
                if opponent_profile is not None:
                    opponent.restore_opponent_profile(opponent_profile)
                opponent.set_game_index(phase)
                counts = {'decisions': 0, 'changed': 0}
                def hook(*, game, policies, choice, decisions):
                    if game.current_player == 1-identity%2:
                        telemetry = policies[1-identity%2].last_decision_telemetry['competitive_switch']
                        counts['decisions'] += 1
                        counts['changed'] += int(telemetry['normal_choice_changed'])
                trace = _game_trace(focal, opponent, seat=identity%2, deal_seed=seed,
                                    recycle_seed=seed+200, starting_player=0, max_decisions=256,
                                    decision_hook=hook)
                rows.append({'identity': identity, 'regime': regime, 'game': gi, 'seed': seed,
                             'termination': trace['termination'], 'outcome': trace['focal_outcome'],
                             'decisions': trace['decisions'],
                             'opponent_activation': counts,
                             'profile_before': before,
                             'profile_after': hashlib.sha256(json.dumps(focal.to_dict(), sort_keys=True).encode()).hexdigest(),
                             'retained': _metric(trace['rows'], 'posterior'),
                             'reset': _metric(trace['rows'], 'reset_shadow')})
                if trace['termination'] != 'win':
                    break
                profile = focal.to_dict()
                opponent_profile = opponent.to_dict()
    for i in range(2):
        prefix = [r for r in rows if r['identity'] == i and r['game'] == 0]
        if len(prefix) != 2 or any(prefix[0][k] != prefix[1][k] for k in
                                   ('termination','outcome','decisions','profile_before','profile_after','retained','reset','opponent_activation')):
            raise AssertionError('stable/switch prefix mismatch')
    if any(hashlib.sha256(Path(f).read_bytes()).hexdigest() != h for f,h in hashes.items()):
        raise RuntimeError('source changed during experiment')
    with path.open('x') as stream:
        json.dump({'kind': 'synthetic_competitive_defense_shadow_pilot_not_promotion',
                   'config': {'identities': 2, 'games': 2, 'decision_cap': 256,
                              'seed_base': seed_base, 'stop_after_incomplete': True},
                   'source_sha256_partial': hashes, 'elapsed_seconds': time.monotonic()-start,
                   'rows': rows}, stream, indent=2)
        stream.write('\n')
    print('rows', len(rows), 'completed', sum(r['termination'] == 'win' for r in rows))


if __name__ == '__main__':
    main()
