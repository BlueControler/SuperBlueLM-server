"""LangGraph Server entrypoint for the Android mobile agent."""

from __future__ import annotations

from .agent import build_agent
from .runtime import phone_gateway, system_gateway

graph = build_agent(phone_gateway, system_gateway)

__all__ = ["graph"]
