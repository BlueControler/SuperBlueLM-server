from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskComplexity = Literal["simple", "complex"]

TEXT_ONLY_REASON = "text_only_answer"
TOOL_REQUIRED_REASON = "tool_or_external_action_required"

PHONE_TOOL_INTENTS = (
    "打开",
    "点击",
    "输入",
    "滑动",
    "返回",
    "搜索框",
    "手机页面",
    "屏幕",
    "应用",
    "app",
    "浏览器",
)

SYSTEM_TOOL_INTENTS = (
    "应用列表",
    "已安装应用",
    "当前位置",
    "日程",
    "提醒",
    "联系人",
    "系统",
)

EXTERNAL_TOOL_INTENTS = (
    "天气",
    "路线",
    "地图",
    "距离",
    "导航",
    "飞书",
    "企业微信",
    "高德",
    "搜索",
    "查询",
    "查一下",
)


@dataclass(frozen=True)
class TaskComplexityResult:
    complexity: TaskComplexity
    track_steps: bool
    reason: str


def classify_task_complexity(text: str) -> TaskComplexityResult:
    normalized = text.strip().lower()
    if not normalized:
        return _simple()

    if _contains_any(normalized, PHONE_TOOL_INTENTS):
        return _complex()
    if _contains_any(normalized, SYSTEM_TOOL_INTENTS):
        return _complex()
    if _contains_any(normalized, EXTERNAL_TOOL_INTENTS):
        return _complex()

    return _simple()


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value.lower() in text for value in values)


def _simple() -> TaskComplexityResult:
    return TaskComplexityResult(
        complexity="simple",
        track_steps=False,
        reason=TEXT_ONLY_REASON,
    )


def _complex() -> TaskComplexityResult:
    return TaskComplexityResult(
        complexity="complex",
        track_steps=True,
        reason=TOOL_REQUIRED_REASON,
    )
