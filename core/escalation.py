"""
Harbor — Escalation detection.
Decides when Harbor should hand off to a human agent.
"""
import re
from config import PersonaConfig


# Default escalation triggers (applies to all personas unless overridden)
DEFAULT_TRIGGERS = [
    "speak to a human",
    "speak to someone",
    "talk to a person",
    "real person",
    "human agent",
    "talk to agent",
    "manager",
    "supervisor",
    "complaint",
    "legal",
    "lawyer",
    "this is urgent",
    "emergency",
]


def should_escalate(message: str, custom_triggers: list[str] | None = None) -> bool:
    """
    Return True if the message contains escalation triggers.
    Checks both default triggers and any persona-specific ones.
    """
    triggers = DEFAULT_TRIGGERS + (custom_triggers or [])
    message_lower = message.lower()
    return any(
        re.search(rf"\b{re.escape(t.lower())}\b", message_lower)
        for t in triggers
    )


def build_escalation_message(persona: PersonaConfig, contact_name: str = "") -> str:
    """Build the handoff message shown to the visitor before escalation."""
    if persona.escalation_prompt:
        return persona.escalation_prompt

    name = f", {contact_name}" if contact_name else ""
    return (
        f"Of course{name} — let me connect you with a member of the "
        f"{persona.business_name} team right now. "
        f"They'll be with you shortly. Please hold on! 👋"
    )
