"""Corrected-rule persistence lifecycle for explicitly enabled account rooms.

The caller must supply a server-verified account UUID, never a browser alias.
Storage implements the existing db.pg consent/snapshot/revision contract.
No research sessions, game transcripts, hidden cards or training data are saved.
"""

from uuid import UUID

from bots.round_policy import RoundAdaptiveBot
# Existing opponent_models.bot_id is VARCHAR(20). The full ruleset identity
# remains mandatory inside the validated profile; never load historical keys.
PROFILE_STORAGE_KEY = "round_adaptive_v2"


class RoundProfileSession:
    """One account, one loaded revision, at most one save per completed game."""

    def __init__(self, verified_user_id, storage):
        self.user_id = str(UUID(verified_user_id)) if verified_user_id else None
        self.storage = storage
        self.status = "not_started"
        self.revision = None
        self._finalized = False

    async def begin_game(self, bot: RoundAdaptiveBot):
        """Load before gameplay; errors must never authorize a fresh overwrite."""
        self._finalized = False
        self.revision = None
        self.status = "anonymous" if self.user_id is None else "not_consented"
        # A new lifecycle must not inherit another identity's account memory,
        # including when its new owner is a guest. Guest room rematches do not
        # call this component; account rematches reload the consented snapshot.
        bot.reset_profile()
        if self.user_id is None:
            return self.status
        # Do not retain a previous persistent identity snapshot if consent has
        # changed or storage cannot be read. Deal-local reset remains separate.
        try:
            preferences = await self.storage.get_privacy_preferences(self.user_id)
            if preferences.get("cross_session_profile") is not True:
                return self.status
            saved, revision = await self.storage.load_opponent_model_snapshot(
                self.user_id, PROFILE_STORAGE_KEY)
            if saved is None:
                if revision is not None:
                    raise ValueError("missing profile has a revision")
                self.status = "new"
            else:
                if type(revision) is not int or revision < 0:
                    raise ValueError("invalid profile revision")
                bot.restore_opponent_profile(saved)
                self.status = "restored"
            self.revision = revision
        except Exception:
            # Fail closed on infrastructure/schema/version errors. The caller
            # can still play, but cannot replace a profile it failed to read.
            self.status = "unavailable"
        return self.status

    async def finish_game(self, bot: RoundAdaptiveBot, *, completed: bool,
                          abandoned: bool = False):
        """Caller supplies verified engine termination, not a client assertion."""
        # Guest rooms may checkpoint interrupted evidence without inventing a
        # completed game. Historical account callers retain completed-only saves.
        if (completed is not True and abandoned is not True) or self._finalized:
            return False
        self._finalized = True
        if self.user_id is None or self.status not in {"new", "restored"}:
            return False
        try:
            # db.pg rechecks current consent inside the save transaction and
            # rejects stale revisions. Never retry with an unconditional save.
            saved = await self.storage.save_opponent_model(
                self.user_id, PROFILE_STORAGE_KEY, bot.to_dict(),
                expected_revision=self.revision)
        except Exception:
            self.status = "save_failed"
            return False
        if saved is not True:
            self.status = "save_failed"
            return False
        self.revision = (self.revision or 0) + 1
        self.status = "saved"
        return True
