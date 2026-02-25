"""
Harbor — Core LangGraph agent.
One agent per conversation turn. Stateless between calls —
conversation history is fetched fresh from Chatwoot each time.
"""
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
import structlog

from config import PersonaConfig, get_settings
from core.escalation import should_escalate, build_escalation_message

log = structlog.get_logger()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    persona: PersonaConfig
    contact_name: str
    escalate: bool
    response: str


def build_agent(persona: PersonaConfig) -> StateGraph:
    """Build a LangGraph agent for a given persona."""
    settings = get_settings()

    # Use GPT-4o for Pro+ clients, local Qwen3-32B for everyone else
    if persona.tier in ("pro", "agency") and settings.harbor_llm_fallback_api_key:
        llm = ChatOpenAI(
            model=settings.harbor_llm_fallback_model,
            api_key=settings.harbor_llm_fallback_api_key,
            temperature=0.7,
        )
    else:
        llm = ChatOpenAI(
            model=settings.harbor_llm_model,
            api_key=settings.harbor_llm_api_key,
            base_url=settings.harbor_llm_base_url,
            temperature=0.7,
        )

    def router(state: AgentState) -> AgentState:
        """Check if we should escalate before even calling LLM."""
        messages = state["messages"]
        if not messages:
            return {**state, "escalate": False}

        last_message = messages[-1].content if messages else ""

        if persona.human_escalation and should_escalate(
            last_message, persona.escalation_triggers
        ):
            log.info("harbor.escalation_triggered", client_id=persona.client_id)
            return {**state, "escalate": True}

        return {**state, "escalate": False}

    def responder(state: AgentState) -> AgentState:
        """Call LLM and generate response."""
        import re
        system = SystemMessage(content=persona.system_prompt)
        response = llm.invoke([system] + state["messages"])
        content = response.content
        # Strip <think>...</think> blocks (Qwen3 chain-of-thought)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        log.info(
            "harbor.response_generated",
            client_id=persona.client_id,
            tokens=response.usage_metadata.get("total_tokens", 0)
            if hasattr(response, "usage_metadata")
            else 0,
        )
        return {**state, "response": content}

    def escalator(state: AgentState) -> AgentState:
        """Build escalation handoff message."""
        msg = build_escalation_message(persona, state.get("contact_name", ""))
        return {**state, "response": msg}

    def route_after_router(state: AgentState) -> str:
        return "escalator" if state.get("escalate") else "responder"

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("responder", responder)
    graph.add_node("escalator", escalator)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_router)
    graph.add_edge("responder", END)
    graph.add_edge("escalator", END)

    return graph.compile()


def history_to_messages(
    raw_messages: list[dict],
) -> list[HumanMessage | AIMessage]:
    """Convert Chatwoot message history to LangChain messages."""
    result = []
    for msg in sorted(raw_messages, key=lambda m: m.get("created_at", 0)):
        content = msg.get("content", "")
        if not content:
            continue
        # message_type: 1 = incoming (visitor), 2 = outgoing (agent/bot)
        msg_type = msg.get("message_type", 1)
        if msg_type == 1:
            result.append(HumanMessage(content=content))
        elif msg_type == 2 and not msg.get("private"):
            result.append(AIMessage(content=content))
    return result
