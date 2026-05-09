"""Shared runtime objects for the LangGraph API process."""

from __future__ import annotations

from .gateways.phone import DeviceGateway
from .gateways.system import SystemToolGateway

phone_gateway = DeviceGateway()
system_gateway = SystemToolGateway()

__all__ = ["phone_gateway", "system_gateway"]
