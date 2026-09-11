import pytest
from analysis.metrics import (
    compute_bluff_calibration_error,
    compute_call_precision,
    compute_shedding_differential,
    compute_adaptation_velocity,
    compute_deception_elo,
)


def test_bce():
    probs = [0.9, 0.1, 0.8]
    labels = [True, False, True]
    # abs(0.9 - 1) = 0.1, abs(0.1 - 0) = 0.1, abs(0.8 - 1) = 0.2 -> avg = 0.4 / 3 = 0.1333
    bce = compute_bluff_calibration_error(probs, labels)
    assert abs(bce - (0.4 / 3)) < 1e-4


def test_call_precision():
    assert compute_call_precision(10, 20) == 50.0
    assert compute_call_precision(0, 0) == 0.0


def test_shedding_differential():
    # Opponent has 18 cards, we have 14 cards -> differential +4
    diff = compute_shedding_differential(14.0, 18.0)
    assert diff == 4.0


def test_adaptation_velocity():
    # Initial prior 0.30, target 0.80, half-target 0.55
    history = [0.30, 0.40, 0.60, 0.75, 0.80]
    t_half = compute_adaptation_velocity(history, 0.80)
    assert t_half == 2  # at index 2, val is 0.60 >= 0.55


def test_deception_elo():
    ratings = {"AcademicBeast": 1500.0, "Honest": 1500.0}
    # AcademicBeast wins, lock hands: Beast=12, Honest=20 (diff=8)
    ra, rb = compute_deception_elo(ratings, "AcademicBeast", "AcademicBeast", "Honest", 12.0, 20.0)
    assert ra > 1500.0
    assert rb < 1500.0
