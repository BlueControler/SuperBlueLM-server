from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from mobile_agent.agent.middleware import DirectPhoneIntentMiddleware


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


def test_launch_with_follow_up_work_keeps_model_planning() -> None:
    middleware = DirectPhoneIntentMiddleware()
    request = _Request(
        messages=[HumanMessage(content="打开飞书并给小王发送消息")],
        state={},
    )
    expected = object()

    response = middleware.wrap_model_call(request, lambda _: expected)

    assert response is expected
