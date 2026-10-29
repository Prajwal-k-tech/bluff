# Bluff research snapshot

This directory contains an isolated research snapshot of the corrected
public-pile-v3 game and experimental adaptive-agent evaluation. It does not
replace the web release in the repository root and is not a deployable update.

After a challenge both players see the ordered pile; only the latest packet
determines its outcome. A pass recycles the pile face down. See `docs/` for the
two rule corrections.

The opening expert combines an empirical response estimate with simulated
opponent hypotheses. The current-deal defense variant and same-claim shadows
are research controls, not validated flagship improvements. Small synthetic
experiments have shown both benefits and failures of retained memory; none
establish human win rate or state-of-the-art strength. In one small switched
opponent pilot the current-deal control lost all four completed post-switch
games, while the baseline won two completed games and two were unstarted
following a timeout. These samples are too small and censored for a reliable
strength comparison.

From this directory, install pytest and run:

```sh
python -m pytest -q tests/test_round_current_deal_defense_bot.py tests/test_round_defense_reset_shadow.py tests/test_round_predecision_bluff_audit.py tests/test_round_public_pile_visibility.py tests/test_round_public_pile_chronology.py
```

Synthetic diagnostic scripts are exploratory, use bounded samples, and must not
be interpreted as human data. Gameplay logs, private profiles, credentials,
generated results, and model weights are not included. The included forecast
audit plan redacts historical human-game observations and is not the original
frozen document.
