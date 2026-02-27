"""
Harbor — Client config loader.
Each client has a persona YAML in personas/examples/<client_id>.yaml
"""
import os
import yaml
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Harbor service
    harbor_secret: str = "dev-secret"
    port: int = 8000

    # Chatwoot connection
    chatwoot_base_url: str = "http://192.168.0.99:30095"
    chatwoot_user_access_token: str = ""  # Super admin token

    # LLM — defaults to local Qwen3-32B (OpenAI-compatible)
    # Named HARBOR_LLM_* to avoid conflict with real OPENAI_API_KEY in Infisical
    harbor_llm_api_key: str = "sk-local"
    harbor_llm_base_url: str = "http://192.168.0.99:8011/v1"
    harbor_llm_model: str = "qwen3-32b"

    # Fallback — GPT-4o (Pro tier clients)
    harbor_llm_fallback_api_key: str = ""
    harbor_llm_fallback_model: str = "gpt-4o"

    # Database — pgvector-enabled Postgres for RAG knowledge base
    database_url: str = "postgresql://sotastack:sotastack-local-2026@localhost:5432/harbor"

    # Redis
    redis_url: str = "redis://localhost:6379/5"

    # Sentry (optional)
    sentry_dsn: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


class PersonaConfig(BaseModel):
    """Per-client persona configuration."""
    client_id: str
    name: str                        # The agent's name (e.g. "Max", "Smokey")
    business_name: str
    business_type: str               # dental, retail, construction, etc.
    system_prompt: str
    greeting: str                    # First proactive message
    escalation_prompt: str           # What to say when handing off to human
    escalation_triggers: list[str]   # Keywords/phrases that trigger escalation
    tools: list[str]                 # Which integrations to load: shopify, cliniko, etc.
    tier: str = "starter"            # starter, growth, pro, agency
    language: str = "en"
    chatwoot_account_id: int = 1
    chatwoot_inbox_id: int = 1
    # Bot identity — Chatwoot agent bot token for this persona
    # Store in Infisical as e.g. HARBOR_BOT_TOKEN_DENTAL_DEMO
    # Messages will appear from the named bot (e.g. "Max") not the admin account
    bot_token_env: str = ""          # env var name that holds the bot token
    # Growth+ only
    human_escalation: bool = False
    chatwoot_escalation_agent_id: int | None = None
    # RAG knowledge base
    rag_enabled: bool = False         # set True to enable retrieval
    rag_client_id: str = ""           # override client_id for KB lookup (empty = use client_id)
    rag_max_chars: int = 3000         # max context chars injected into prompt
    # Pro+ only
    proactive_triggers: bool = False
    multi_channel: bool = False

    @property
    def bot_token(self) -> str:
        """Look up the bot token from env at runtime."""
        if not self.bot_token_env:
            return ""
        import os
        return os.environ.get(self.bot_token_env, "")


PERSONAS_DIR = Path(__file__).parent / "personas" / "examples"


@lru_cache
def load_persona(client_id: str) -> PersonaConfig:
    """Load a client's persona config from YAML."""
    config_path = PERSONAS_DIR / f"{client_id}.yaml"
    if not config_path.exists():
        raise ValueError(f"No persona config found for client_id: {client_id}")
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return PersonaConfig(**data)


def get_all_personas() -> list[PersonaConfig]:
    """Load all persona configs from the personas directory."""
    personas = []
    for yaml_file in PERSONAS_DIR.glob("*.yaml"):
        try:
            personas.append(load_persona(yaml_file.stem))
        except Exception:
            pass
    return personas
