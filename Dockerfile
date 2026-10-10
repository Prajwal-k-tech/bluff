FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/next.config.ts frontend/postcss.config.mjs frontend/tsconfig.json ./
COPY frontend/src/ ./src/
COPY frontend/public/fonts/cormorant-garamond-italic-latin.woff2 frontend/public/fonts/OFL-CormorantGaramond.txt ./public/fonts/
ENV BLUFF_STATIC_EXPORT=1 NEXT_PUBLIC_API_URL=same-origin
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt
COPY cards.py game.py game_v2.py server_v2.py web_app.py profile_auth.py guest_identity.py web_origins.py ./
COPY bots/__init__.py bots/base.py bots/prob.py bots/round_observation.py bots/round_policy.py bots/round_card_memory.py bots/round_profile_session.py ./bots/
COPY db/__init__.py db/pg.py db/guest.py db/schema.sql ./db/
COPY --from=frontend /build/out/ ./web/
COPY --from=frontend /build/public/fonts/ ./web/fonts/
ENV PORT=8000 BLUFF_WEB_DIR=/app/web BLUFF_ENABLE_ACCOUNT_PROFILES=0
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn web_app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
