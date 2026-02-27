"""
Harbor — Core LangGraph agent.
One agent per conversation turn. Stateless between calls —
conversation history is fetched fresh from Chatwoot each time.

RAG-enhanced: retrieves relevant knowledge chunks before calling LLM.
"""
import re
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
import structlog

from config import PersonaConfig, get_settings
from core.escalation import should_escalate, build_escalation_message

log = structlog.get_logger()

# Lazy singleton — initialized on first use
_knowledge_service = None


async def _get_knowledge_service():
    """Get or create the knowledge service singleton."""
    global _knowledge_service
    if _knowledge_service is None:
        from services.knowledge import KnowledgeService

        settings = get_settings()
        _knowledge_service = KnowledgeService(settings.database_url)
        await _knowledge_service.initialize()
        log.info("harbor.knowledge_service.initialized")
    return _knowledge_service


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    persona: PersonaConfig
    contact_name: str
    escalate: bool
    response: str
    rag_context: str  # injected product/knowledge context


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

    async def retriever(state: AgentState) -> AgentState:
        """Retrieve relevant knowledge chunks for the latest message."""
        messages = state["messages"]
        if not messages or not persona.rag_enabled:
            return {**state, "rag_context": ""}

        last_message = messages[-1].content if messages else ""
        try:
            ks = await _get_knowledge_service()
            context = await ks.get_context(
                persona.rag_client_id or persona.client_id,
                last_message,
                max_chars=persona.rag_max_chars,
            )
            if context:
                log.info(
                    "harbor.rag.context_retrieved",
                    client_id=persona.client_id,
                    chars=len(context),
                )
            return {**state, "rag_context": context}
        except Exception as e:
            log.error("harbor.rag.retrieval_failed", error=str(e))
            return {**state, "rag_context": ""}

    def responder(state: AgentState) -> AgentState:
        """Call LLM with persona prompt + RAG context."""
        # Build system prompt with RAG context injected
        system_content = persona.system_prompt
        rag_context = state.get("rag_context", "")
        if rag_context:
            system_content += (
                "\n\n## Relevant Product/Knowledge Context\n"
                "Use this information to answer the customer's question accurately. "
                "Only reference products listed here — do not invent products or prices.\n\n"
                f"{rag_context}"
            )

        system = SystemMessage(content=system_content)
        response = llm.invoke([system] + state["messages"])
        content = response.content
        # Strip <think>...</think> blocks (Qwen3 chain-of-thought)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        log.info(
            "harbor.response_generated",
            client_id=persona.client_id,
            rag=bool(rag_context),
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
        return "escalator" if state.get("escalate") else "retriever"

    # Build graph: router → retriever → responder (or escalator)
    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("retriever", retriever)
    graph.add_node("responder", responder)
    graph.add_node("escalator", escalator)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_router)
    graph.add_edge("retriever", "responder")
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
        # Chatwoot message_type: 0 = incoming (visitor), 1 = outgoing (agent/bot)
        msg_type = msg.get("message_type", 0)
        if msg_type == 0:
            result.append(HumanMessage(content=content))
        elif msg_type == 1 and not msg.get("private"):
            result.append(AIMessage(content=content))
    return result
