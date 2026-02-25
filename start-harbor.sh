#!/bin/bash
# Harbor — Start script with Infisical secret injection
# Fetches all HARBOR_* secrets from Infisical at runtime
# Pattern: same as ~/bin/start-litellm.sh

set -e

HARBOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTH_FILE="$HOME/.config/infisical/agent-auth.json"
PROJECT_ID="628dc743-e40f-4267-82ec-78841aae7b6b"
INFISICAL_URL="https://infisical.sotastack.com.au"
ENV="dev"
PORT="${PORT:-8025}"

echo "[harbor] Fetching secrets from Infisical..."

# Authenticate as javis-agent
CLIENT_ID=$(python3 -c "import json; print(json.load(open('$AUTH_FILE'))['clientId'])")
CLIENT_SECRET=$(python3 -c "import json; print(json.load(open('$AUTH_FILE'))['clientSecret'])")

_TOKEN=$(curl -sf "$INFISICAL_URL/api/v1/auth/universal-auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"clientId\":\"$CLIENT_ID\",\"clientSecret\":\"$CLIENT_SECRET\"}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['accessToken'])")

# Helper: fetch a single secret by key name
get_secret() {
  curl -sf \
    "$INFISICAL_URL/api/v3/secrets/raw/$1?workspaceId=$PROJECT_ID&environment=$ENV" \
    -H "Authorization: Bearer $_TOKEN" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['secret']['secretValue'])"
}

# Inject secrets as env vars
export CHATWOOT_USER_ACCESS_TOKEN=$(get_secret "CHATWOOT_USER_ACCESS_TOKEN")
export CHATWOOT_BASE_URL=$(get_secret "CHATWOOT_BASE_URL")
export HARBOR_SECRET=$(get_secret "HARBOR_SECRET")
export HARBOR_LLM_API_KEY=$(get_secret "HARBOR_LLM_API_KEY")
export HARBOR_LLM_BASE_URL=$(get_secret "HARBOR_LLM_BASE_URL")
export HARBOR_LLM_MODEL=$(get_secret "HARBOR_LLM_MODEL")

# Clear token from env — don't leave it hanging around
unset _TOKEN

echo "[harbor] Secrets loaded. Starting on port $PORT..."

cd "$HARBOR_DIR"
exec python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT"
