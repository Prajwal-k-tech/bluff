import pytest
from bots.archetype_classifier import ArchetypeClassifier


def test_archetype_initial_uniform():
    clf = ArchetypeClassifier()
    posts = clf.infer_posteriors(0, 0, 0, 0)
    for p in posts.values():
        assert abs(p - 0.25) < 1e-4


def test_classify_honest_rock():
    clf = ArchetypeClassifier()
    # 0 bluffs, 6 honest plays, 1 call, 5 passes
    top, conf = clf.top_archetype(bluffs=0, honest=6, calls=1, passes=5)
    assert top == "Honest_Rock"
    assert conf > 0.60
    ab, bb, ac, bc = clf.get_calibrated_prior(0, 6, 1, 5)
    # Mean bluff rate should be low (<0.10)
    mean_b = ab / (ab + bb)
    assert mean_b < 0.15


def test_classify_calling_station():
    clf = ArchetypeClassifier()
    # 1 bluff, 3 honest plays, 9 calls, 1 pass
    top, conf = clf.top_archetype(bluffs=1, honest=3, calls=9, passes=1)
    assert top == "Calling_Station"
    assert conf > 0.80
    ab, bb, ac, bc = clf.get_calibrated_prior(1, 3, 9, 1)
    # Mean call rate should be high (>0.70)
    mean_c = ac / (ac + bc)
    assert mean_c > 0.70


def test_classify_hyper_maniac():
    clf = ArchetypeClassifier()
    # 8 bluffs, 2 honest plays, 2 calls, 3 passes
    top, conf = clf.top_archetype(bluffs=8, honest=2, calls=2, passes=3)
    assert top == "Hyper_Maniac"
    assert conf > 0.80
    ab, bb, ac, bc = clf.get_calibrated_prior(8, 2, 2, 3)
    # Mean bluff rate should be high (>0.60)
    mean_b = ab / (ab + bb)
    assert mean_b > 0.60
