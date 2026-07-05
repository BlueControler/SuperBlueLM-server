from __future__ import annotations

from typing import Any, cast

from deepagents import create_deep_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState

from ..gateways.phone import DeviceGateway
from ..gateways.system import SystemToolGateway
from ..local_model_runtime import build_cloud_model
from ..tools.completion import create_completion_tools
from ..tools.external import create_external_tools
from ..tools.memory import create_memory_tools
from ..tools.phone import create_phone_tools
from ..tools.scenario_system import create_scenario_system_tools
from ..tools.system import create_system_tools
from .middleware import (
    ModeToolAccessMiddleware,
    ResetAgentRunStateMiddleware,
    RouteModelMiddleware,
    RoutedSystemPromptMiddleware,
    SyncPhoneStateMiddleware,
    TaskComplexityMiddleware,
)
from .medical_travel_sop import (
    MedicalTravelSopMiddleware,
    MedicalTravelSopRunner,
    build_medical_travel_sop_runner,
)
from .meeting_minutes_sop import (
    MeetingMinutesSopMiddleware,
    MeetingMinutesSopRunner,
    build_meeting_minutes_sop_runner,
)
from .risk_gate import HighRiskActionGateMiddleware
from .scenario3_demo import Scenario3DemoMiddleware
from .scenarios.app_inventory_skill import AppInventoryQueryMiddleware
from .scenarios.open_app_skill import OpenAppSkillMiddleware
from .scenarios.weather_advice_skill import WeatherAdviceMiddleware
from .trace_middleware import TraceMiddleware


def build_middleware_stack(
    *,
    phone_gateway: DeviceGateway,
    system_gateway: SystemToolGateway,
    phone_tool_names: set[str],
    device_scoped_tool_names: set[str],
    meeting_minutes_runner: MeetingMinutesSopRunner | None = None,
    medical_travel_runner: MedicalTravelSopRunner | None = None,
) -> list[AgentMiddleware[AgentState[Any], None, Any]]:
    """Returns the security-sensitive middleware order used by the main agent."""
    return cast(
        list[AgentMiddleware[AgentState[Any], None, Any]],
        [
            ResetAgentRunStateMiddleware(),
            TraceMiddleware(),
            OpenAppSkillMiddleware(phone_gateway, system_gateway),
            AppInventoryQueryMiddleware(system_gateway),
            WeatherAdviceMiddleware(),
            MeetingMinutesSopMiddleware(meeting_minutes_runner),
            Scenario3DemoMiddleware(),
            MedicalTravelSopMiddleware(medical_travel_runner),
            HighRiskActionGateMiddleware(),
            ModeToolAccessMiddleware(phone_tool_names, device_scoped_tool_names),
            TaskComplexityMiddleware(),
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
    scenario_system_tools = create_scenario_system_tools(system_gateway)
    external_tools = create_external_tools()
    device_scoped_tool_names = (
        phone_tool_names
        | {tool.name for tool in system_tools}
        | {tool.name for tool in scenario_system_tools}
    )
    middleware = build_middleware_stack(
        phone_gateway=phone_gateway,
        system_gateway=system_gateway,
        phone_tool_names=phone_tool_names,
        device_scoped_tool_names=device_scoped_tool_names,
        meeting_minutes_runner=build_meeting_minutes_sop_runner(
            scenario_system_tools=scenario_system_tools,
        ),
        medical_travel_runner=build_medical_travel_sop_runner(
            external_tools=external_tools,
            system_tools=system_tools,
        ),
    )
    return create_deep_agent(
        model=main_cloud_model,
        tools=[
            *phone_tools,
            *system_tools,
            *scenario_system_tools,
            *external_tools,
            *create_memory_tools(),
            *create_completion_tools(),
        ],
        system_prompt="",
        middleware=middleware,
    )
