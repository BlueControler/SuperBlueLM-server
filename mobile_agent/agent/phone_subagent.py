from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, Protocol, cast

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.runtime import Runtime
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from ..gateways.phone import DeviceGateway
from ..prompt_assets import PHONE_SUBAGENT_SYSTEM_PROMPT
from ..tools.phone import create_phone_tools
from .middleware import SyncPhoneStateMiddleware

PhoneTodoStatus = Literal[
    "completed",
    "failed",
    "needs_user_action",
    "budget_exhausted",
]

BASE64_IMAGE_PATTERN = re.compile(
    r"data:image/[^,\s]+;base64,[A-Za-z0-9+/=\r\n]+",
    re.IGNORECASE,
)
UI_TAG_PATTERN = re.compile(r"<[^>]+>")
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    (["']?(?:
        token|secret|key|password|passwd|authorization|credential|
        cookie|session(?:_id)?|access_token|api_key
    )["']?\s*[:=]\s*)
    (?:
        ["'][^"']*["']|
        [^,\s;]+
    )
    """
)
BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+")
AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"(?i)(authorization\s*:\s*)(?:basic|bearer)\s+[^,\s;]+"
)
COOKIE_HEADER_PATTERN = re.compile(r"(?i)(cookie\s*:\s*)[^\r\n]+")


class PhoneStateSummary(BaseModel):
    current_package: str | None = Field(alias="currentPackage")
    activity: str | None
    has_screenshot: bool = Field(alias="hasScreenshot")
    has_ui: bool = Field(alias="hasUi")

    model_config = {"populate_by_name": True}


class PhoneSubagentDecision(BaseModel):
    status: Literal["completed", "failed", "needs_user_action"]
    summary: str
    needs_main_agent_plan: bool = Field(alias="needsMainAgentPlan")
    error: str | None = None

    model_config = {"populate_by_name": True}


class PhoneTodoExecution(BaseModel):
    status: PhoneTodoStatus
    todo: str
    summary: str
    phone_state: PhoneStateSummary = Field(alias="phoneState")
    tool_call_count: int = Field(alias="toolCallCount")
    needs_main_agent_plan: bool = Field(alias="needsMainAgentPlan")
    error: str | None = None

    model_config = {"populate_by_name": True}


class PhoneToolBudgetState(AgentState[PhoneSubagentDecision], total=False):
    phone_tool_call_count: NotRequired[Annotated[int, PrivateStateAttr]]


class PhoneToolBudgetExceededError(RuntimeError):
    def __init__(self, *, executed_count: int, limit: int) -> None:
        super().__init__(f"Phone tool budget exhausted after {executed_count}/{limit} calls.")
        self.executed_count = executed_count
        self.limit = limit


class PhoneToolSequenceError(RuntimeError):
    pass


class PhoneToolBudgetMiddleware(
    AgentMiddleware[PhoneToolBudgetState, None, PhoneSubagentDecision]
):
    state_schema = PhoneToolBudgetState

    def __init__(self, phone_tool_names: set[str], limit: int) -> None:
        self.phone_tool_names = frozenset(phone_tool_names)
        self.limit = max(limit, 1)

    def after_model(
        self,
        state: PhoneToolBudgetState,
        runtime: Runtime[None],
    ) -> dict[str, int] | None:
        phone_calls = self._latest_phone_calls(state)
        if not phone_calls:
            return None
        if len(phone_calls) > 1:
            raise PhoneToolSequenceError(
                "Phone tools must be called sequentially so each action uses fresh UI state."
            )

        executed_count = state.get("phone_tool_call_count", 0)
        if executed_count >= self.limit:
            raise PhoneToolBudgetExceededError(
                executed_count=executed_count,
                limit=self.limit,
            )
        return {"phone_tool_call_count": executed_count + 1}

    async def aafter_model(
        self,
        state: PhoneToolBudgetState,
        runtime: Runtime[None],
    ) -> dict[str, int] | None:
        return self.after_model(state, runtime)

    def _latest_phone_calls(self, state: PhoneToolBudgetState) -> list[ToolCall]:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                return [
                    call
                    for call in message.tool_calls
                    if call["name"] in self.phone_tool_names
                ]
        return []


class InvokableAgent(Protocol):
    async def ainvoke(self, state: dict[str, Any]) -> Mapping[str, Any]: ...


AgentFactory = Callable[..., InvokableAgent]


class PhoneSubagentRunner:
    def __init__(
        self,
        phone_gateway: DeviceGateway,
        model: BaseChatModel | str,
        *,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        self.phone_gateway = phone_gateway
        self.model = model
        self.agent_factory = agent_factory or cast(AgentFactory, create_agent)

    async def execute(
        self,
        todo: str,
        *,
        allow_short_chain: bool,
    ) -> PhoneTodoExecution:
        if phone_text_contains_sensitive_data(todo):
            logger.warning("Phone TODO contains sensitive data and requires user takeover.")
            return self._needs_user_action(todo, "phone_todo_contains_sensitive_data")

        budget = _tool_budget(allow_short_chain)
        tools = create_phone_tools(self.phone_gateway)
        phone_tool_names = {tool.name for tool in tools}
        agent = self.agent_factory(
            model=self.model,
            tools=tools,
            system_prompt=PHONE_SUBAGENT_SYSTEM_PROMPT,
            middleware=[
                SyncPhoneStateMiddleware(self.phone_gateway),
                PhoneToolBudgetMiddleware(phone_tool_names, budget),
            ],
            response_format=ToolStrategy(PhoneSubagentDecision),
        )
        try:
            state = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"当前 TODO: {todo}\n"
                                f"allow_short_chain={str(allow_short_chain).lower()}\n"
                                f"最多调用 {budget} 次手机工具。"
                            )
                        )
                    ]
                }
            )
        except PhoneToolBudgetExceededError as exc:
            logger.warning("Phone subagent stopped because its tool budget was exhausted.")
            return self._budget_exhausted(todo, exc.executed_count, exc.limit)
        except PhoneToolSequenceError:
            logger.warning("Phone subagent attempted parallel UI tool calls.")
            return self._failed(todo, "phone_parallel_tool_calls_rejected")
        except Exception:
            logger.exception("Phone subagent execution failed.")
            return self._failed(todo, "phone_subagent_execution_failed")

        try:
            decision = PhoneSubagentDecision.model_validate(state["structured_response"])
        except Exception:
            logger.exception("Phone subagent returned an invalid structured response.")
            return self._failed(todo, "phone_subagent_invalid_response")
        return PhoneTodoExecution(
            status=decision.status,
            todo=redact_phone_text(todo),
            summary=redact_phone_text(decision.summary),
            phoneState=self._phone_state_summary(),
            toolCallCount=_count_phone_tool_calls(
                state.get("messages", ()),
                phone_tool_names,
            ),
            needsMainAgentPlan=decision.needs_main_agent_plan,
            error=redact_phone_text(decision.error) if decision.error else None,
        )

    def _phone_state_summary(self) -> PhoneStateSummary:
        try:
            session = self.phone_gateway.get_session()
        except Exception:
            return PhoneStateSummary(
                currentPackage=None,
                activity=None,
                hasScreenshot=False,
                hasUi=False,
            )

        info = session.device_info
        return PhoneStateSummary(
            currentPackage=info.current_package if info else None,
            activity=info.activity if info else None,
            hasScreenshot=bool(info and info.screenshot),
            hasUi=bool(info and info.ui),
        )

    def _budget_exhausted(
        self,
        todo: str,
        executed_count: int,
        limit: int,
    ) -> PhoneTodoExecution:
        return PhoneTodoExecution(
            status="budget_exhausted",
            todo=redact_phone_text(todo),
            summary=f"Phone TODO stopped after reaching the {limit}-tool budget.",
            phoneState=self._phone_state_summary(),
            toolCallCount=executed_count,
            needsMainAgentPlan=True,
            error="phone_tool_budget_exhausted",
        )

    def _failed(self, todo: str, error: str) -> PhoneTodoExecution:
        return PhoneTodoExecution(
            status="failed",
            todo=redact_phone_text(todo),
            summary="Phone subagent execution failed.",
            phoneState=self._phone_state_summary(),
            toolCallCount=0,
            needsMainAgentPlan=True,
            error=error,
        )


    def _needs_user_action(self, todo: str, error: str) -> PhoneTodoExecution:
        return PhoneTodoExecution(
            status="needs_user_action",
            todo=redact_phone_text(todo),
            summary="Phone TODO requires user takeover because it contains sensitive data.",
            phoneState=self._phone_state_summary(),
            toolCallCount=0,
            needsMainAgentPlan=True,
            error=error,
        )


def _tool_budget(allow_short_chain: bool) -> int:
    env_name = (
        "PHONE_SUBAGENT_SHORT_CHAIN_MAX_TOOL_CALLS"
        if allow_short_chain
        else "PHONE_SUBAGENT_MAX_TOOL_CALLS"
    )
    default = "4" if allow_short_chain else "1"
    try:
        return max(int(os.getenv(env_name, default)), 1)
    except ValueError:
        return int(default)


def _count_phone_tool_calls(
    messages: Sequence[Any],
    phone_tool_names: set[str],
) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, ToolMessage) and message.name in phone_tool_names
    )


def redact_phone_text(value: str) -> str:
    without_images = BASE64_IMAGE_PATTERN.sub("data:image/***;base64,***", value)
    without_ui_tags = UI_TAG_PATTERN.sub("<ui-redacted>", without_images)
    without_authorization = AUTHORIZATION_HEADER_PATTERN.sub(
        r"\1***",
        without_ui_tags,
    )
    without_cookies = COOKIE_HEADER_PATTERN.sub(r"\1***", without_authorization)
    without_bearer = BEARER_PATTERN.sub(r"\1***", without_cookies)
    return SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\1***", without_bearer)


def phone_text_contains_sensitive_data(value: str) -> bool:
    return redact_phone_text(value) != value


__all__ = [
    "PhoneStateSummary",
    "PhoneSubagentRunner",
    "PhoneTodoExecution",
    "PhoneTodoStatus",
    "phone_text_contains_sensitive_data",
    "redact_phone_text",
]
