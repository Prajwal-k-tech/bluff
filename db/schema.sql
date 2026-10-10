-- Bluff Bot — Neon Postgres schema (Phase 2: web data pipeline)
-- Apply with:  neonctl sql --file db/schema.sql   (or any SQL client)
--
-- Web source: human-vs-bot games played via the FastAPI backend.
-- Records the same fields as the terminal JSONL format (docs/data-pipeline.md)
-- so both sources merge into one unified dataset for paper analysis.

-- Users keyed by a deterministic UUID derived from a verified Clerk subject.
-- The current backend does not store the raw Clerk subject in clerk_user_id.
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clerk_user_id VARCHAR(255) UNIQUE, -- reserved for an explicit account-link migration
    display_name VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Purpose-specific opt-ins. Missing rows mean both purposes are off.
CREATE TABLE IF NOT EXISTS user_privacy_preferences (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    cross_session_profile BOOLEAN NOT NULL DEFAULT FALSE,
    research_logging BOOLEAN NOT NULL DEFAULT FALSE,
    consent_version VARCHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per game
CREATE TABLE IF NOT EXISTS game_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID REFERENCES users(id),
    bot_mode VARCHAR(20) NOT NULL,      -- 'random' | 'honest' | 'cardcount' | 'bayesian' | 'purenn' | 'hybrid'
    result VARCHAR(10) NOT NULL,        -- 'win' | 'loss' | 'draw'
    num_turns INTEGER,
    duration_seconds INTEGER,
    finished_at TIMESTAMPTZ,            -- set by log_session_end; absent in the
                                        -- original architecture.md draft → added
                                        -- here to match db/pg.py UPDATE.
    model_loaded BOOLEAN DEFAULT FALSE, -- persisted opponent model was loaded at
                                        -- session start (claim-(b) ablation switch:
                                        -- win-rate delta loaded vs cold-start for the
                                        -- SAME user — docs/s3-design.md §4).
    policy_provenance JSONB,             -- static effective policy after restore;
                                        -- audit-only, NULL for historical games
    research_consent_version VARCHAR(64), -- non-NULL only for explicitly opted-in logs
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Every play / call / pass in every game
CREATE TABLE IF NOT EXISTS actions (
    id BIGSERIAL PRIMARY KEY,
    game_id UUID REFERENCES game_sessions(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    player_type VARCHAR(10) NOT NULL,   -- 'human' | 'bot'
    action_type VARCHAR(10) NOT NULL,   -- 'play' | 'call' | 'pass'
    cards_played JSONB,                 -- ["3H", "3S"]
    claimed_rank INTEGER,               -- 2..14 (Rank enum value)
    was_bluff BOOLEAN,
    bluff_called BOOLEAN DEFAULT FALSE,
    caller_was_right BOOLEAN,
    hand_size INTEGER,
    opponent_hand_size INTEGER,
    pile_size INTEGER,
    p_bluff_estimate FLOAT,             -- bot's Bayesian P(bluff) at decision time
    selected_action_probability DOUBLE PRECISION,
    selected_action_probability_kind VARCHAR(16), -- marginal | conditional | deterministic | unknown
    bluff_head_state_vector JSONB,      -- legal pre-response hybrid-v1-42 detector input
    bluff_head_schema VARCHAR(32),
    bluff_head_conflict SMALLINT,
    bluff_head_aux JSONB,               -- opponent mean packet size and current delta
    decision_ms INTEGER,                -- HUMAN decision latency (ms from prompt to
                                        -- action). Timing tells predict bluffs (Bitan &
                                        -- Kraus: response duration is a top feature).
                                        -- NULL for bot rows / terminal imports. Server
                                        -- fills this when S3 wires web logging.
    state_vector JSONB,                  -- pre-decision public-information encoding
    legal_mask JSONB,                    -- legal action mask at that decision
    action_index INTEGER,                -- 54-action encoding used by BluffNet
    state_schema VARCHAR(32),             -- e.g. state-v1-39; prevents silent drift
    claim_id UUID,                       -- same UUID on play and its response;
                                         -- audit only, NULL for historical rows
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Serialized Bayesian opponent model per (user, bot) pair — cross-game learning
CREATE TABLE IF NOT EXISTS opponent_models (
    user_id UUID REFERENCES users(id),
    bot_id VARCHAR(20) NOT NULL,        -- bot_mode string
    model_data JSONB NOT NULL,          -- OpponentModel.to_dict() output
    games_played INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, bot_id)
);

-- Schema v2 patch (2026-09-10): add model_loaded to existing deployments.
-- Idempotent — safe to re-run; new deployments get it from CREATE TABLE above.
ALTER TABLE game_sessions ADD COLUMN IF NOT EXISTS model_loaded BOOLEAN DEFAULT FALSE;
ALTER TABLE game_sessions ADD COLUMN IF NOT EXISTS policy_provenance JSONB;
ALTER TABLE game_sessions ADD COLUMN IF NOT EXISTS research_consent_version VARCHAR(64);
ALTER TABLE actions ADD COLUMN IF NOT EXISTS state_vector JSONB;
ALTER TABLE actions ADD COLUMN IF NOT EXISTS legal_mask JSONB;
ALTER TABLE actions ADD COLUMN IF NOT EXISTS action_index INTEGER;
ALTER TABLE actions ADD COLUMN IF NOT EXISTS state_schema VARCHAR(32);
-- 2026-10-01: explicit play/response audit linkage; no historical backfill.
ALTER TABLE actions ADD COLUMN IF NOT EXISTS claim_id UUID;
ALTER TABLE actions ADD COLUMN IF NOT EXISTS selected_action_probability DOUBLE PRECISION;
ALTER TABLE actions ADD COLUMN IF NOT EXISTS selected_action_probability_kind VARCHAR(16);
-- 2026-10-08: detector training view; separate from the 39-feature action imitation snapshot.
ALTER TABLE actions ADD COLUMN IF NOT EXISTS bluff_head_state_vector JSONB;
ALTER TABLE actions ADD COLUMN IF NOT EXISTS bluff_head_schema VARCHAR(32);
ALTER TABLE actions ADD COLUMN IF NOT EXISTS bluff_head_conflict SMALLINT;
ALTER TABLE actions ADD COLUMN IF NOT EXISTS bluff_head_aux JSONB;

-- Indexes (from docs/architecture.md)
CREATE INDEX IF NOT EXISTS idx_actions_session ON actions(game_id);
CREATE INDEX IF NOT EXISTS idx_actions_player_type ON actions(player_type);
CREATE INDEX IF NOT EXISTS idx_sessions_player ON game_sessions(player_id);
CREATE INDEX IF NOT EXISTS idx_sessions_bot_mode ON game_sessions(bot_mode);
