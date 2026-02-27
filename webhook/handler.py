"""
Harbor — Chatwoot webhook handler.
Receives all Chatwoot events, filters to visitor messages only,
routes to the right LangGraph agent per client_id.

Routing: supports both URL-based (/webhook/{client_id}) and
inbox-based routing (inbox_id → client_id mapping from persona configs).
"""
import asyncio
import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from config import load_persona, get_all_personas
from core.agent import AgentState, build_agent, history_to_messages
from integrations.chatwoot import ChatwootClient

log = structlog.get_logger()
router = APIRouter()


# ---------------------------------------------------------------------------
# Inbox → Persona routing
# ---------------------------------------------------------------------------

_inbox_map: dict[int, str] | None = None


def _build_inbox_map() -> dict[int, str]:
    """Build inbox_id → client_id mapping from all persona configs."""
    global _inbox_map
    if _inbox_map is None:
        _inbox_map = {}
        for persona in get_all_personas():
            if persona.chatwoot_inbox_id:
                _inbox_map[persona.chatwoot_inbox_id] = persona.client_id
        log.info("harbor.inbox_map_built", mapping=_inbox_map)
    return _inbox_map


def resolve_client_id(client_id: str, payload: dict) -> str:
    """
    Resolve the actual client_id for this webhook event.
    Priority: inbox_id mapping > URL client_id fallback.
    """
    # inbox can be at top level, nested in inbox{}, or in conversation{}
    inbox_id = (
        payload.get("inbox", {}).get("id")
        or payload.get("inbox_id")
        or payload.get("conversation", {}).get("inbox_id")
    )
    if inbox_id:
        inbox_map = _build_inbox_map()
        mapped = inbox_map.get(inbox_id)
        if mapped:
            return mapped
    return client_id


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@router.post("/webhook/{client_id}")
async def chatwoot_webhook(
    client_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Chatwoot posts every conversation event here.
    We filter to incoming visitor messages only, then process async.
    """
    payload = await request.json()
    event_type = payload.get("event")

    # Resolve actual client_id from inbox mapping
    resolved_id = resolve_client_id(client_id, payload)

    log.debug("harbor.webhook_received", client_id=resolved_id, evt=event_type,
              url_client_id=client_id,
              inbox_id=payload.get("inbox", {}).get("id"),
              msg_type=payload.get("message_type") if event_type == "message_created" else None)

    # Handle new conversations — send proactive greeting
    if event_type == "conversation_created":
        conversation = payload.get("conversation", {})
        conversation_id = conversation.get("id")
        account_id = payload.get("account", {}).get("id")
        if conversation_id and account_id:
            background_tasks.add_task(
                send_greeting,
                client_id=resolved_id,
                account_id=account_id,
                conversation_id=conversation_id,
            )
        return {"status": "greeting_queued"}

    # Only care about new incoming messages
    if event_type != "message_created":
        return {"status": "ignored", "reason": "not a message event"}

    # Chatwoot sends message_created with fields at the TOP level of the payload
    msg_type = payload.get("message_type")
    is_incoming = msg_type == 0 or msg_type == "incoming"
    if not is_incoming:
        return {"status": "ignored", "reason": f"not an incoming visitor message (type={msg_type})"}

    # Ignore empty messages
    content = payload.get("content", "").strip()
    if not content:
        return {"status": "ignored", "reason": "empty message"}

    # Extract conversation context
    conversation = payload.get("conversation", {})
    conversation_id = conversation.get("id")
    account_id = payload.get("account", {}).get("id")
    contact = payload.get("contact", payload.get("sender", {}))

    if not conversation_id or not account_id:
        raise HTTPException(status_code=400, detail="Missing conversation or account id")

    # Process in background — return 200 immediately to Chatwoot
    background_tasks.add_task(
        process_message,
        client_id=resolved_id,
        account_id=account_id,
        conversation_id=conversation_id,
        content=content,
        contact=contact,
    )

    return {"status": "queued", "client_id": resolved_id}


# ---------------------------------------------------------------------------
# Greeting — sent when a new conversation is created
# ---------------------------------------------------------------------------

async def send_greeting(client_id: str, account_id: int, conversation_id: int):
    """Send persona greeting when a visitor starts a new conversation."""
    try:
        persona = load_persona(client_id)
    except ValueError:
        return
    chatwoot = ChatwootClient(account_id=account_id, bot_token=persona.bot_token)
    await chatwoot.send_message(conversation_id, persona.greeting)
    log.info("harbor.greeting_sent", client_id=client_id, conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

async def process_message(
    client_id: str,
    account_id: int,
    conversation_id: int,
    content: str,
    contact: dict,
):
    """
    Load persona → fetch history → run RAG retriever → run agent → send response back.
    """
    try:
        persona = load_persona(client_id)
    except ValueError as e:
        log.error("harbor.unknown_client", client_id=client_id, error=str(e))
        return

    chatwoot = ChatwootClient(account_id=account_id, bot_token=persona.bot_token)
    contact_name = contact.get("name", "")

    # Fetch conversation history for context
    raw_messages = await chatwoot.get_conversation_messages(conversation_id)
    history = history_to_messages(raw_messages)

    # Run LangGraph agent
    agent = build_agent(persona)
    initial_state = AgentState(
        messages=history,
        persona=persona,
        contact_name=contact_name,
        escalate=False,
        response="",
        rag_context="",
    )

    # Agent graph has async nodes (retriever), use ainvoke
    result = await agent.ainvoke(initial_state)
    response_text = result.get("response", "")
    escalate = result.get("escalate", False)

    if not response_text:
        log.warning("harbor.empty_response", client_id=client_id, conversation_id=conversation_id)
        return

    # Send the response message
    await chatwoot.send_message(conversation_id, response_text)

    # If escalating: leave private note for agent + assign to human
    if escalate and persona.human_escalation:
        await chatwoot.send_private_note(
            conversation_id,
            f"🤖 Harbor escalated this conversation.\n"
            f"Visitor said: \"{content}\"\n"
            f"Context: {len(history)} messages exchanged.",
        )
        if persona.chatwoot_escalation_agent_id:
            await chatwoot.assign_agent(
                conversation_id, persona.chatwoot_escalation_agent_id
            )

    log.info(
        "harbor.message_processed",
        client_id=client_id,
        conversation_id=conversation_id,
        escalated=escalate,
    )
