from __future__ import annotations

from typing import Any, cast

from deepagents import create_deep_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState

from ..gateways.phone import DeviceGateway
from ..gateways.system import SystemToolGateway
from ..local_model_runtime import build_cloud_model, build_phone_subagent_model
from ..tools.external import create_external_tools
from ..tools.phone import create_phone_tools
from ..tools.system import create_system_tools
from .middleware import (
    DirectPhoneIntentMiddleware,
    ModeToolAccessMiddleware,
    RouteModelMiddleware,
    RoutedSystemPromptMiddleware,
    SyncPhoneStateMiddleware,
    TaskComplexityMiddleware,
)
from .phone_delegation import ResetPhoneTodoMiddleware, create_phone_delegation_tool
from .phone_subagent import PhoneSubagentRunner
from .risk_gate import HighRiskActionGateMiddleware
from .trace_middleware import TraceMiddleware


def build_middleware_stack(
    *,
    phone_gateway: DeviceGateway,
    phone_tool_names: set[str],
    device_scoped_tool_names: set[str],
) -> list[AgentMiddleware[AgentState[Any], None, Any]]:
    """Returns the security-sensitive middleware order used by the main agent."""
    return cast(
        list[AgentMiddleware[AgentState[Any], None, Any]],
        [
            ResetPhoneTodoMiddleware(),
            TraceMiddleware(),
            HighRiskActionGateMiddleware(),
            ModeToolAccessMiddleware(phone_tool_names, device_scoped_tool_names),
            TaskComplexityMiddleware(),
            DirectPhoneIntentMiddleware(),
            RouteModelMiddleware(),
            RoutedSystemPromptMiddleware(),
            SyncPhoneStateMiddleware(phone_gateway),
        ],
    )


def build_agent(phone_gateway: DeviceGateway, system_gateway: SystemToolGateway):
    main_cloud_model = build_cloud_model()
    phone_tools = create_phone_tools(phone_gateway)
    phone_tool_names = {tool.name for tool in phone_tools}
    system_tools = create_system_tools(system_gateway)
    device_scoped_tool_names = phone_tool_names | {tool.name for tool in system_tools}
    phone_subagent_model = build_phone_subagent_model(main_cloud_model)
    phone_delegation_tool = create_phone_delegation_tool(
        PhoneSubagentRunner(phone_gateway, phone_subagent_model)
    )
    middleware = build_middleware_stack(
        phone_gateway=phone_gateway,
        phone_tool_names=phone_tool_names,
        device_scoped_tool_names=device_scoped_tool_names,
    )
    return create_deep_agent(
        model=main_cloud_model,
        tools=[
            *phone_tools,
            phone_delegation_tool,
            *system_tools,
            *create_external_tools(),
        ],
        system_prompt="",
        middleware=middleware,
    )
