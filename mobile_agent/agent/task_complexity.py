from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskComplexity = Literal["simple", "complex"]

TEXT_ONLY_REASON = "text_only_answer"
ZERO_OR_ONE_TOOL_REASON = "zero_or_one_tool_call"
MULTI_STEP_REASON = "multi_step_plan_required"

MULTI_STEP_ACTION_INTENTS = (
    "观察",
    "打开",
    "点击",
    "输入",
    "滑动",
    "返回",
    "搜索",
    "进入",
    "提交",
    "停在",
    "切换",
    "等待",
    "启动",
)

SINGLE_TOOL_HINTS = (
    "手机页面",
    "屏幕",
    "应用列表",
    "已安装应用",
    "当前位置",
    "日程",
    "提醒",
    "联系人",
    "系统",
    "天气",
    "路线",
    "地图",
    "距离",
    "导航",
    "飞书",
    "企业微信",
    "高德",
    "查询",
    "查一下",
    "读取",
    "获取",
)

TOOL_INTENT_CATEGORIES = (
    ("应用列表", "已安装应用"),
    ("当前位置", "定位"),
    ("日程",),
    ("提醒",),
    ("联系人",),
    ("天气",),
    ("路线", "地图", "距离", "导航"),
    ("飞书",),
    ("企业微信",),
)


@dataclass(frozen=True)
class TaskComplexityResult:
    complexity: TaskComplexity
    track_steps: bool
    reason: str


def classify_task_complexity(text: str) -> TaskComplexityResult:
    normalized = text.strip().lower()
    if not normalized:
        return _text_only()

    matched_actions = _matched_values(normalized, MULTI_STEP_ACTION_INTENTS)
    tool_category_count = _count_matching_groups(normalized, TOOL_INTENT_CATEGORIES)

    if len(matched_actions) >= 2:
        return _complex()
    if tool_category_count >= 2:
        return _complex()
    if tool_category_count >= 1 and _has_phone_action(matched_actions):
        return _complex()

    if matched_actions or tool_category_count >= 1:
        return _zero_or_one_tool()
    if _contains_any(normalized, SINGLE_TOOL_HINTS):
        return _zero_or_one_tool()

    return _text_only()


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value.lower() in text for value in values)


def _matched_values(text: str, values: tuple[str, ...]) -> list[str]:
    return [value for value in values if value.lower() in text]


def _count_matching_groups(text: str, groups: tuple[tuple[str, ...], ...]) -> int:
    return sum(1 for group in groups if _contains_any(text, group))


def _has_phone_action(matched_actions: list[str]) -> bool:
    return any(action not in {"观察", "搜索"} for action in matched_actions)


def _text_only() -> TaskComplexityResult:
    return TaskComplexityResult(
        complexity="simple",
        track_steps=False,
        reason=TEXT_ONLY_REASON,
    )


def _zero_or_one_tool() -> TaskComplexityResult:
    return TaskComplexityResult(
        complexity="simple",
        track_steps=False,
        reason=ZERO_OR_ONE_TOOL_REASON,
    )


def _complex() -> TaskComplexityResult:
    return TaskComplexityResult(
        complexity="complex",
        track_steps=True,
        reason=MULTI_STEP_REASON,
    )
