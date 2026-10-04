import random

import pytest

from test_bots import seed_tournament_rngs


def test_tournament_seed_resets_python_and_torch_rngs():
    torch = pytest.importorskip("torch")

    seed_tournament_rngs(20261005)
    first_python = random.random()
    first_torch = torch.rand(8)

    seed_tournament_rngs(20261005)
    assert random.random() == first_python
    assert torch.equal(torch.rand(8), first_torch)
