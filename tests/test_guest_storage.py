import json
import asyncio
import uuid

import pytest

from db import guest


USER = uuid.uuid4()
OTHER = uuid.uuid4()
GAME = uuid.uuid4()
VERSION = "guest-consent-v1"
PROVENANCE = {"policy": "flagship", "ruleset": "bluff-fixed-rounds-v2"}
EVENT = {"player_type": "human", "action_type": "play", "cards": ["2H"]}


def run(coro):
    return asyncio.run(coro)


class Store:
    def __init__(self, *, consent=True, owner=USER, fail=False):
        self.consent = consent
        self.owner = owner
        self.fail = fail
        self.events = {}
        self.deletes = []
        self.finished = None

    def acquire(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def transaction(self):
        return self

    async def fetchrow(self, query, *args):
        if self.fail:
            raise OSError("database failed")
        if "user_privacy_preferences" in query:
            return {"research_logging": self.consent,
                    "consent_version": VERSION if self.consent else "old"}
        if "SELECT result, num_turns" in query:
            if self.finished is None:
                return {"result": "started", "num_turns": None,
                        "duration_seconds": None, "finished_at": None}
            return {"result": self.finished[0], "num_turns": self.finished[1],
                    "duration_seconds": self.finished[2], "finished_at": "set"}
        if "SELECT id, finished_at" in query:
            return {"id": GAME, "finished_at": "set" if self.finished else None}
        if "FROM game_sessions WHERE id=$1" in query:
            return {"player_id": self.owner,
                    "bot_mode": "flagship",
                    "research_consent_version": VERSION,
                    "policy_provenance": PROVENANCE}
        if "SELECT id FROM game_sessions" in query:
            return {"id": GAME} if self.owner == args[1] else None
        if "SELECT state_vector FROM actions" in query:
            return self.events.get((args[0], args[1], args[2]))
        return None

    async def execute(self, query, *args):
        if self.fail:
            raise OSError("database failed")
        if query.startswith("DELETE FROM"):
            self.deletes.append((query, args))
            return "DELETE 1" if query.endswith("users WHERE id=$1") else "DELETE 0"
        if "INSERT INTO actions" in query:
            game_id, index, player, action, schema, serialized = args
            self.events[(game_id, index, schema)] = {"state_vector": json.loads(serialized)}
        if "UPDATE game_sessions SET result" in query:
            self.finished = (args[2], args[3], args[4])
        return "INSERT 0 1"


@pytest.fixture
def store(monkeypatch):
    instance = Store()
    async def get_pool():
        return instance
    monkeypatch.setattr(guest.pg, "get_pool", get_pool)
    return instance


def test_start_game_is_noop_without_matching_consent(monkeypatch):
    instance = Store(consent=False)
    async def get_pool(): return instance
    monkeypatch.setattr(guest.pg, "get_pool", get_pool)
    assert not run(guest.start_game(str(USER), "flagship", str(GAME), PROVENANCE, VERSION))


def test_start_game_does_not_claim_another_users_session(monkeypatch):
    instance = Store(owner=OTHER)
    async def get_pool(): return instance
    monkeypatch.setattr(guest.pg, "get_pool", get_pool)
    assert not run(guest.start_game(str(USER), "flagship", str(GAME), PROVENANCE, VERSION))


def test_event_is_idempotent_but_conflicting_duplicate_fails(store):
    assert run(guest.append_event(str(USER), str(GAME), 2, EVENT))
    assert run(guest.append_event(str(USER), str(GAME), 2, EVENT))
    with pytest.raises(ValueError, match="different data"):
        run(guest.append_event(str(USER), str(GAME), 2,
                               {**EVENT, "cards": ["3H"]}))


def test_delete_data_targets_only_the_requested_user(store):
    assert run(guest.delete_data(str(USER)))
    assert len(store.deletes) == 3
    assert all(args == (USER,) for _query, args in store.deletes)


def test_finish_game_is_idempotent_and_rejects_conflicting_finalization(store):
    assert run(guest.finish_game(str(USER), str(GAME), "win", 10, 30))
    assert run(guest.finish_game(str(USER), str(GAME), "win", 10, 30))
    with pytest.raises(ValueError, match="different results"):
        run(guest.finish_game(str(USER), str(GAME), "loss", 11, 31))


def test_finished_game_accepts_event_retry_but_rejects_new_event(store):
    assert run(guest.append_event(str(USER), str(GAME), 2, EVENT))
    assert run(guest.finish_game(str(USER), str(GAME), "win", 3, 10))
    assert run(guest.append_event(str(USER), str(GAME), 2, EVENT))
    assert not run(guest.append_event(str(USER), str(GAME), 3, EVENT))


def test_storage_failures_propagate(monkeypatch):
    instance = Store(fail=True)
    async def get_pool(): return instance
    monkeypatch.setattr(guest.pg, "get_pool", get_pool)
    with pytest.raises(OSError, match="database failed"):
        run(guest.start_game(str(USER), "flagship", str(GAME), PROVENANCE, VERSION))
