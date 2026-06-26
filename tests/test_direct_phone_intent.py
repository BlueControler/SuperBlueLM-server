from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage

from mobile_agent.agent.middleware import DirectPhoneIntentMiddleware
from mobile_agent.agent.phone_delegation import (
    ResetPhoneTodoMiddleware,
    create_phone_delegation_tool,
)
from mobile_agent.agent.phone_subagent import PhoneTodoExecution


class _FailingBindableFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, *args: object, **kwargs: object) -> "_FailingBindableFakeChatModel":
        del args, kwargs
        return self

    def _generate(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("model must not run for a direct app launch")


class _Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def execute(
        self,
        todo: str,
        *,
        allow_short_chain: bool,
        device_id: str | None = None,
        trace_parent_id: str | None = None,
    ) -> PhoneTodoExecution:
        del device_id, trace_parent_id
        self.calls.append((todo, allow_short_chain))
        return PhoneTodoExecution(
            status="completed",
            todo=todo,
            summary="已发起打开应用操作。",
            phoneState={
                "currentPackage": "com.ss.android.lark",
                "activity": ".main.app.MainActivity",
                "hasScreenshot": True,
                "hasUi": True,
            },
            toolCallCount=1,
            needsMainAgentPlan=False,
        )


@dataclass(frozen=True)
class _Request:
    messages: list[Any]
    state: dict[str, Any]

    def override(self, **changes: Any) -> "_Request":
        return replace(self, **changes)


def test_simple_feishu_launch_bypasses_model_and_delegates_known_package() -> None:
    middleware = DirectPhoneIntentMiddleware()
    request = _Request(
        messages=[HumanMessage(content="请用你的手机工具帮我打开飞书")],
        state={},
    )

    response = middleware.wrap_model_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("model must not run")),
    )

    message = response.result[0]
    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "execute_phone_todo"
    assert message.tool_calls[0]["args"] == {
        "todo": "打开飞书（包名 com.ss.android.lark）",
        "allow_short_chain": False,
    }


def test_completed_direct_launch_returns_final_summary_without_redelegating() -> None:
    middleware = DirectPhoneIntentMiddleware()
    request = _Request(
        messages=[HumanMessage(content="打开飞书app")],
        state={
            "phone_todo_steps": (
                {
                    "index": 1,
                    "progressKey": "phone-todo-1",
                    "name": "打开飞书（包名 com.ss.android.lark）",
                    "status": "completed",
                    "summary": "已发起打开应用操作。",
                },
            )
        },
    )

    response = middleware.wrap_model_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("model must not run")),
    )

    message = response.result[0]
    assert isinstance(message, AIMessage)
    assert message.content == "已发起打开应用操作。"
    assert message.tool_calls == []


def test_direct_launch_agent_stops_after_phone_todo_completion() -> None:
    runner = _Runner()
    agent = create_deep_agent(
        model=_FailingBindableFakeChatModel(responses=[]),
        tools=[create_phone_delegation_tool(runner)],
        middleware=[
            ResetPhoneTodoMiddleware(),
            DirectPhoneIntentMiddleware(),
        ],
    )

    result = asyncio.run(
        asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": "打开飞书app"}]}),
            timeout=3,
        )
    )

    assert runner.calls == [("打开飞书（包名 com.ss.android.lark）", False)]
    assert result["messages"][-1].content == "已发起打开应用操作。"
    assert sum(isinstance(message, ToolMessage) for message in result["messages"]) == 1


def test_launch_with_follow_up_work_keeps_model_planning() -> None:
    middleware = DirectPhoneIntentMiddleware()
    request = _Request(
        messages=[HumanMessage(content="打开飞书并给小王发送消息")],
        state={},
    )
    expected = object()

    response = middleware.wrap_model_call(request, lambda _: expected)

    assert response is expected
