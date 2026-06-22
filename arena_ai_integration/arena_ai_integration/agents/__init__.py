"""Agent factory for arena_ai_integration."""

from __future__ import annotations

from arena_ai_integration.agents.base_agent import BaseAgent
from arena_ai_integration.agents.lelan_agent import LeLanAgent
from arena_ai_integration.agents.socialnav_agent import SocialNavAgent
from arena_ai_integration.agents.urbannav_agent import UrbanNavAgent

AGENT_REGISTRY = {
    'socialnav': SocialNavAgent,
    'urbannav': UrbanNavAgent,
    'lelan': LeLanAgent,
}


def create_agent(agent_type: str) -> BaseAgent:
    key = (agent_type or 'socialnav').strip().lower()
    if key not in AGENT_REGISTRY:
        supported = ', '.join(sorted(AGENT_REGISTRY))
        raise ValueError(f"Unknown agent_type '{agent_type}'. Supported: {supported}")
    return AGENT_REGISTRY[key]()
