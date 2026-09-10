-- Bluff Bot — Neon Postgres schema (Phase 2: web data pipeline)
-- Apply with:  neonctl sql --file db/schema.sql   (or any SQL client)
--
-- Web source: human-vs-bot games played via the FastAPI backend.
-- Records the same fields as the terminal JSONL format (docs/data-pipeline.md)
-- so both sources merge into one unified dataset for paper analysis.

-- Users (mirror of Clerk user ids — no passwords stored here; Clerk handles auth)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clerk_user_id VARCHAR(255) UNIQUE,
    display_name VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per game
CREATE TABLE IF NOT EXISTS game_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID REFERENCES users(id),
    bot_mode VARCHAR(20) NOT NULL,      -- 'random' | 'honest' | 'cardcount' | 'bayesian' | 'purenn' | 'hybrid'
    result VARCHAR(10) NOT NULL,        -- 'win' | 'loss' | 'draw'
    num_turns INTEGER,
    duration_seconds INTEGER,
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

-- Indexes (from docs/architecture.md)
CREATE INDEX IF NOT EXISTS idx_actions_session ON actions(game_id);
CREATE INDEX IF NOT EXISTS idx_actions_player_type ON actions(player_type);
CREATE INDEX IF NOT EXISTS idx_sessions_player ON game_sessions(player_id);
CREATE INDEX IF NOT EXISTS idx_sessions_bot_mode ON game_sessions(bot_mode);
