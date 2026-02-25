"""
Harbor — Chatwoot webhook handler.
Receives all Chatwoot events, filters to visitor messages only,
routes to the right LangGraph agent per client_id.
"""
import asyncio
import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from config import load_persona
from core.agent import AgentState, build_agent, history_to_messages
from integrations.chatwoot import ChatwootClient

log = structlog.get_logger()
router = APIRouter()


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
    log.debug("harbor.webhook_received", client_id=client_id, evt=event_type)

    # Handle new conversations — send Harbor's proactive greeting
    if event_type == "conversation_created":
        conversation = payload.get("conversation", {})
        conversation_id = conversation.get("id")
        account_id = payload.get("account", {}).get("id")
        if conversation_id and account_id:
            background_tasks.add_task(
                send_greeting,
                client_id=client_id,
                account_id=account_id,
                conversation_id=conversation_id,
            )
        return {"status": "greeting_queued"}

    # Only care about new incoming messages
    if event_type != "message_created":
        return {"status": "ignored", "reason": "not a message event"}

    message = payload.get("message", {})

    # Chatwoot message_type: 0 = incoming (visitor), 1 = outgoing (agent/bot), 2 = activity
    if message.get("message_type") != 0:
        return {"status": "ignored", "reason": "not an incoming visitor message"}

    # Ignore empty messages
    content = message.get("content", "").strip()
    if not content:
        return {"status": "ignored", "reason": "empty message"}

    # Extract conversation context
    conversation = payload.get("conversation", {})
    conversation_id = conversation.get("id")
    account_id = payload.get("account", {}).get("id")
    contact = payload.get("contact", {})

    if not conversation_id or not account_id:
        raise HTTPException(status_code=400, detail="Missing conversation or account id")

    # Process in background — return 200 immediately to Chatwoot
    background_tasks.add_task(
        process_message,
        client_id=client_id,
        account_id=account_id,
        conversation_id=conversation_id,
        content=content,
        contact=contact,
    )

    return {"status": "queued"}


# ---------------------------------------------------------------------------
# Greeting — sent when a new conversation is created
# ---------------------------------------------------------------------------

async def send_greeting(client_id: str, account_id: int, conversation_id: int):
    """Send Max's persona greeting when a visitor starts a new conversation."""
    try:
        persona = load_persona(client_id)
    except ValueError:
        return
    chatwoot = ChatwootClient(account_id=account_id)
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
    Load persona → fetch history → run agent → send response back.
    Handles escalation by assigning to a human agent in Chatwoot.
    """
    try:
        persona = load_persona(client_id)
    except ValueError as e:
        log.error("harbor.unknown_client", client_id=client_id, error=str(e))
        return

    chatwoot = ChatwootClient(account_id=account_id)
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
    )

    result = await asyncio.to_thread(agent.invoke, initial_state)
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
