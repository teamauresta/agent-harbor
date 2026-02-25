# Harbor 🚢

**AI web chat agent framework — Chatwoot + LangGraph**

Harbor gives any business an AI-powered chat widget that actually knows their business.
Built on Chatwoot (MIT) as the platform layer, with LangGraph as the AI brain.

> Self-hosted alternative to Intercom + AI. $0 platform cost. Your LLM, your data.

---

## Architecture

```
Client website
    └── Chatwoot widget embed
            │
            ▼
    Chatwoot (self-hosted)     ← platform: widget, inbox, human handoff
            │ webhook
            ▼
    Harbor service             ← brain: LangGraph agent, persona, tools
            │
            ├── Local LLM (Qwen3-32B)   ← $0/call
            └── GPT-4o (Pro tier)       ← optional fallback
```

**Rule:** We deploy Chatwoot, we never modify its source. Harbor is a separate service.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/teamauresta/agent-harbor
cd agent-harbor

# 2. Install deps
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your Chatwoot URL + token + LLM endpoint

# 4. Add a client persona
cp personas/examples/dental-demo.yaml personas/examples/my-client.yaml
# Edit the YAML with your client's details

# 5. Run
python main.py
```

---

## Adding a Client

1. Create `personas/examples/{client_id}.yaml` (see examples)
2. In Chatwoot: create an account + inbox for the client
3. Set the inbox webhook URL to: `https://your-harbor-url/webhook/{client_id}`
4. Done — Harbor handles all conversations for that inbox

---

## Tiers

| Tier | Features |
|------|----------|
| Starter | AI-only, 500 conversations/mo |
| Growth | + Human escalation, proactive triggers, 1 integration |
| Pro | + Multi-channel, 3 integrations, custom persona |
| Agency | White-label, unlimited, all integrations |

---

## Deploying Chatwoot (K3s)

See `~/k3s-manifests/chatwoot/README.md`

## Deploying Harbor (Railway)

```bash
railway login
railway init
railway up
```

Set environment variables in the Railway dashboard from `.env.example`.

---

## Stack

- **FastAPI** — webhook server
- **LangGraph** — agent orchestration
- **Chatwoot** — chat platform (MIT, self-hosted)
- **Qwen3-32B** — default LLM (local, OpenAI-compatible)
- **GPT-4o** — optional Pro/Agency tier LLM

---

Built by [SOTAStack](https://sotastack.com.au)
