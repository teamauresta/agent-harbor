"""
Harbor — Chatwoot API client.
Handles all communication back to Chatwoot: sending messages,
escalating to human agents, resolving conversations.
"""
import httpx
import structlog
from config import get_settings

log = structlog.get_logger()


class ChatwootClient:
    def __init__(self, account_id: int):
        settings = get_settings()
        self.base_url = settings.chatwoot_base_url
        self.account_id = account_id
        self.headers = {
            "api_access_token": settings.chatwoot_user_access_token,
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1/accounts/{self.account_id}/{path}"

    async def send_message(
        self,
        conversation_id: int,
        message: str,
        message_type: str = "outgoing",
        private: bool = False,
    ) -> dict:
        """Send a message to a conversation."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._url(f"conversations/{conversation_id}/messages"),
                headers=self.headers,
                json={
                    "content": message,
                    "message_type": message_type,
                    "private": private,
                },
                timeout=10,
            )
            resp.raise_for_status()
            log.info("chatwoot.message_sent", conversation_id=conversation_id)
            return resp.json()

    async def send_private_note(self, conversation_id: int, note: str) -> dict:
        """Send an internal note (visible to agents only, not the customer)."""
        return await self.send_message(conversation_id, note, private=True)

    async def assign_agent(
        self, conversation_id: int, agent_id: int
    ) -> dict:
        """Assign conversation to a human agent (escalation)."""
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                self._url(f"conversations/{conversation_id}/assignments"),
                headers=self.headers,
                json={"assignee_id": agent_id},
                timeout=10,
            )
            resp.raise_for_status()
            log.info(
                "chatwoot.agent_assigned",
                conversation_id=conversation_id,
                agent_id=agent_id,
            )
            return resp.json()

    async def set_status(
        self, conversation_id: int, status: str = "resolved"
    ) -> dict:
        """Update conversation status: open, resolved, pending, snoozed."""
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                self._url(f"conversations/{conversation_id}/update"),
                headers=self.headers,
                json={"status": status},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_conversation_messages(
        self, conversation_id: int
    ) -> list[dict]:
        """Fetch full message history for a conversation."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self._url(f"conversations/{conversation_id}/messages"),
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("payload", {}).get("messages", [])

    async def get_contact(self, contact_id: int) -> dict:
        """Fetch contact details."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self._url(f"contacts/{contact_id}"),
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

    async def update_contact(
        self, contact_id: int, attributes: dict
    ) -> dict:
        """Update contact details (name, email, phone, custom attributes)."""
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                self._url(f"contacts/{contact_id}"),
                headers=self.headers,
                json=attributes,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
