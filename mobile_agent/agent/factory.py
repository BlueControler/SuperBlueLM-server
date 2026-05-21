from __future__ import annotations

from typing import Any, cast

from deepagents import create_deep_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState

from ..gateways.phone import DeviceGateway
from ..gateways.system import SystemToolGateway
from ..local_model_runtime import build_cloud_model
from ..tools.external import create_external_tools
from ..tools.phone import create_phone_tools
from ..tools.system import create_system_tools
from .middleware import (
    RouteModelMiddleware,
    RoutedSystemPromptMiddleware,
    SyncPhoneStateMiddleware,
    TaskComplexityMiddleware,
)


def build_agent(phone_gateway: DeviceGateway, system_gateway: SystemToolGateway):
    middleware = cast(
        list[AgentMiddleware[AgentState[Any], None, Any]],
        [
            TaskComplexityMiddleware(),
            RouteModelMiddleware(),
            RoutedSystemPromptMiddleware(),
            SyncPhoneStateMiddleware(phone_gateway),
        ],
    )
    return create_deep_agent(
        model=build_cloud_model(),
        tools=[
            *create_phone_tools(phone_gateway),
            *create_system_tools(system_gateway),
            *create_external_tools(),
        ],
        system_prompt="",
        middleware=middleware,
    )
