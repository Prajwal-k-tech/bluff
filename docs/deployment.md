# Deployment Guide — Bluff (Render + Vercel + Neon)

> Status: everything creds-free is verified (render.yaml valid, CPU torch pinned,
> server boots with all 7 tiers, 53 tests green, browser E2E green). The steps
> below need YOUR accounts/keys — nothing here can run without them.
> Tooling already installed on this box: `vercel` 59.16.0, `neonctl` 4.17.1.

## 0. Accounts & keys you need (all have free tiers)

| Service | Get it at | What you take away |
|---|---|---|
| Neon (Postgres) | https://console.neon.tech → New Project (`bluff`) | Connection string (`DATABASE_URL`) |
| Render (backend) | https://dashboard.render.com → New → Blueprint | Service URL (`https://bluff-backend-XXXX.onrender.com`) |
| Vercel (frontend) | https://vercel.com → Add New Project | Production URL |
| Clerk (auth) | https://dashboard.clerk.com → your app → API Keys | **Production** publishable key (dev keys show a warning overlay — seen in testing) |

Authenticate the CLIs once: `vercel login` (browser flow), `neonctl auth` (browser flow).

## 1. Push the repo to GitHub (Render deploys from the repo)

```bash
git push origin main   # Render Blueprint reads render.yaml from here
```

## 2. Neon: create DB + apply schema (~5 min)

```bash
# connection string (paste into a local .env, NEVER commit it)
export DATABASE_URL="$(neonctl connection-string --project-id <your-project-id>)"
# apply schema (needs psql; or paste db/schema.sql into Neon's SQL Editor)
psql "$DATABASE_URL" -f db/schema.sql
# verify
psql "$DATABASE_URL" -c "\dt"   # expect: users, game_sessions, actions, opponent_models
```

## 3. Render: launch the backend (~10 min + first build)

Dashboard → New → **Blueprint** → select the repo (`render.yaml` is at root).
Then set env vars on the `bluff-backend` service:
- `DATABASE_URL` = the Neon string from step 2 (**secret**, dashboard-only)
- `BLUFF_NN_CHECKPOINT` = leave default (`nn/checkpoints/final.pt`)

Notes:
- Plan is `free`, 1 worker — pinned deliberately (game rooms are in-memory; do NOT scale past 1 instance).
- **512MB RAM check (Gate-3 finding):** torch loads at startup (~250-350MB) + FastAPI (~80MB) — tight but fits. If Render OOM-kills on cold start, that's the first thing to look at.
- Verify: `curl https://<your-render-url>/bots` → all 7 tiers listed.

## 4. Checkpoint vendoring (do AFTER the v8 verdict lands)

`nn/checkpoints/` is gitignored, so a fresh deploy serves random-play PureNN (default bot is `bayesian`, fully playable regardless). Once the deployment checkpoint is chosen:
```bash
git add -f nn/checkpoints/final.pt   # ~356KB, negligible
git commit -m "vendor deployment checkpoint" && git push origin main  # Render auto-redeploys
```

## 5. Frontend: Vercel (~5 min)

```bash
cd frontend
vercel --prod
```
Then set env vars (Vercel dashboard → Project → Settings → Environment Variables, then redeploy):
- `NEXT_PUBLIC_API_URL` = `https://<your-render-url>` (no trailing slash)
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` = production Clerk key (replaces dev keys)

## 6. Pre-launch checklist (Gate-3 findings — do these before announcing)

- [ ] `server.py` CORS: `allow_origins=["*"]` → tighten to your Vercel URL (required once Clerk is live)
- [ ] Clerk production keys (dev keys show a warning overlay)
- [ ] Render cold start completes without OOM (watch first deploy logs)
- [ ] Play one full game on the Vercel URL; confirm rows appear in Neon's `game_sessions` table
- [ ] Confirm the draw-lock / draw-heavy behavior is as documented (expect draws in honest-heavy matchups — by design, see game-rules §4)

## Rollback / constraints

- In-memory rooms: any backend restart kills active games (documented, `architecture.md` §3).
- Free tiers sleep when idle (first request after idle is slow — normal).
