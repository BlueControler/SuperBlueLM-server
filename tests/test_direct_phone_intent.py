from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool

from mobile_agent.agent.middleware import DirectPhoneIntentMiddleware


class _FailingBindableFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, *args: object, **kwargs: object) -> "_FailingBindableFakeChatModel":
        del args, kwargs
        return self

    def _generate(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("model must not run for a direct app launch")


@dataclass(frozen=True)
class _Request:
    messages: list[Any]
    state: dict[str, Any]

    def override(self, **changes: Any) -> "_Request":
        return replace(self, **changes)


def test_simple_feishu_launch_bypasses_model_and_calls_launch_tool() -> None:
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
    assert message.tool_calls[0]["name"] == "launch"
    assert message.tool_calls[0]["args"] == {"package": "com.ss.android.lark"}


def test_completed_direct_launch_returns_final_summary_without_model() -> None:
    middleware = DirectPhoneIntentMiddleware()
    request = _Request(
        messages=[HumanMessage(content="打开飞书app")],
        state={
            "messages": [
                HumanMessage(content="打开飞书app"),
                ToolMessage(
                    content='{"ok":true,"currentPackage":"com.ss.android.lark"}',
                    tool_call_id="direct-launch",
                    name="launch",
                    status="success",
                ),
            ]
        },
    )

    response = middleware.wrap_model_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("model must not run")),
    )

    message = response.result[0]
    assert isinstance(message, AIMessage)
    assert message.content == "已发起打开飞书。"
    assert message.tool_calls == []


def test_direct_launch_agent_stops_after_launch_completion() -> None:
    calls: list[str] = []

    @tool("launch", description="Launch an Android app by package name.")
    async def launch(package: str) -> str:
        calls.append(package)
        return '{"ok":true,"currentPackage":"com.ss.android.lark"}'

    agent = create_deep_agent(
        model=_FailingBindableFakeChatModel(responses=[]),
        tools=[launch],
        middleware=[DirectPhoneIntentMiddleware()],
    )

    result = asyncio.run(
        asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": "打开飞书app"}]}),
            timeout=3,
        )
    )

    assert calls == ["com.ss.android.lark"]
    assert result["messages"][-1].content == "已发起打开飞书。"
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
