# Bluff playable release

This release snapshot serves the approved fixed-rank two-player game and four
bots: Pranjol (adaptive prototype), Sanja (Math), Sudhnashu (Honest), and Tanmoy
(Random). It contains no trained weights, research datasets or experimental
search policies. Bot strength against humans is not yet established.

The rules are recorded in `docs/RULES_CORRECTION_2026-10-10.md`.
Math uses approximate card beliefs, fixed bluff/call priors of .30/.35 and
short-horizon card utility; it is not mathematically optimal. Pranjol adds
smoothed contextual opponent evidence, not proven human-level play.

## Run

Build `Dockerfile.web` (named `Dockerfile` in the standalone release branch):

```sh
docker build -f Dockerfile.web -t bluff-web .
docker run --rm -p 8000:8000 bluff-web
```

On the standalone branch use `docker build -t bluff-web .` instead. Visit
`http://localhost:8000`. The same origin serves frontend, `/api` HTTP/WebSocket
endpoints and `/health`. Use one instance and one worker; rooms are in-memory.

For HTTPS hosting set `BLUFF_CORS_ALLOWED_ORIGINS` to the exact public origin,
`BLUFF_COOKIE_SECURE=1`, and `BLUFF_ENABLE_ACCOUNT_PROFILES=0`. Use health check
`/health`. Never commit secrets or put database credentials in frontend config.

## Guest memory and data

No account or Clerk setup is required. A random HttpOnly cookie identifies a
returning browser. Data settings separately opt into persistent opponent memory
and research logging; both start off. Research snapshots can contain private
game state and must never be exposed to opponents or public clients.

Without `DATABASE_URL`, play works and room rematches retain behavioral memory,
but server-side durable profiles and research logs are unavailable. Losing the
cookie loses access to saved identity. Disconnecting ends a room, not a resumable
deal. The Data panel supports revocation and deletion.

Database deployment is a separate reviewed step: `db/schema.sql` must match the
target database, and retention/access controls must be established before
enabling collection. The app does not silently migrate production on startup.
Do not advertise persistent cross-session learning until verified against the
actual hosted database.

## Verification and limitations

The development workspace's selected corrected-game regression passed 357 tests
with one database-dependent check skipped before this packaging change.
Release-specific build and HTTP/WebSocket results are recorded in the shared
development handoff; a local build is not proof of successful public hosting.
The complete research history stays in the development workspace, not this
minimal deployment snapshot. Old-rule training results do not validate this game.

Free hosting can sleep after inactivity and restart active rooms. A public
demo does not imply an always-on service or measured human strength.

The bundled Cormorant Garamond font's license is in
`frontend/public/fonts/OFL-CormorantGaramond.txt`.
