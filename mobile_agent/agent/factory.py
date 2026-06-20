from __future__ import annotations

from typing import Any, cast

from deepagents import create_deep_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState

from ..gateways.phone import DeviceGateway
from ..gateways.system import SystemToolGateway
from ..local_model_runtime import build_cloud_model, build_phone_subagent_model
from ..tools.completion import create_completion_tools
from ..tools.external import create_external_tools
from ..tools.memory import create_memory_tools
from ..tools.phone import create_phone_tools
from ..tools.scenario_system import create_scenario_system_tools
from ..tools.system import create_system_tools
from .middleware import (
    ModeToolAccessMiddleware,
    RouteModelMiddleware,
    RoutedSystemPromptMiddleware,
    SyncPhoneStateMiddleware,
    TaskComplexityMiddleware,
)
from .phone_delegation import ResetPhoneTodoMiddleware, create_phone_delegation_tool
from .phone_subagent import PhoneSubagentRunner


def build_agent(phone_gateway: DeviceGateway, system_gateway: SystemToolGateway):
    main_cloud_model = build_cloud_model()
    phone_tools = create_phone_tools(phone_gateway)
    phone_tool_names = {tool.name for tool in phone_tools}
    system_tools = create_system_tools(system_gateway)
    scenario_system_tools = create_scenario_system_tools(system_gateway)
    device_scoped_tool_names = (
        phone_tool_names
        | {tool.name for tool in system_tools}
        | {tool.name for tool in scenario_system_tools}
    )
    phone_subagent_model = build_phone_subagent_model(main_cloud_model)
    phone_delegation_tool = create_phone_delegation_tool(
        PhoneSubagentRunner(phone_gateway, phone_subagent_model)
    )
    middleware = cast(
        list[AgentMiddleware[AgentState[Any], None, Any]],
        [
            ResetPhoneTodoMiddleware(),
            ModeToolAccessMiddleware(phone_tool_names, device_scoped_tool_names),
            TaskComplexityMiddleware(),
            RouteModelMiddleware(),
            RoutedSystemPromptMiddleware(),
            SyncPhoneStateMiddleware(phone_gateway),
        ],
    )
    return create_deep_agent(
        model=main_cloud_model,
        tools=[
            *phone_tools,
            phone_delegation_tool,
            *system_tools,
            *scenario_system_tools,
            *create_external_tools(),
            *create_memory_tools(),
            *create_completion_tools(),
        ],
        system_prompt="",
        middleware=middleware,
    )
