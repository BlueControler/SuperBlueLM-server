from __future__ import annotations

import asyncio
import difflib
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..gateways.system import SystemGatewayError, SystemToolGateway
from ..json_types import JsonObject, to_json_value
from ..progress import emit_task_complexity, emit_task_progress
from .middleware import _message_content_to_text
from .state import MobileAgentState, device_id_from_mapping

AppQueryKind = Literal["exact_app", "category", "capability", "custom"]
AppType = Literal["all", "third", "system"]

APP_INVENTORY_PHASE = "app_inventory_query"
APP_INVENTORY_TASK_TITLE = "应用列表检索"
TOTAL_STEPS = 4

APP_CATEGORY_REGISTRY: dict[str, JsonObject] = {
    "browser": {
        "label": "浏览器类应用",
        "keywords": ["浏览器", "browser", "web", "网页", "上网", "chrome", "edge", "firefox", "brave", "opera", "uc", "夸克", "via"],
        "package_keywords": ["browser", "chrome", "firefox", "edge", "brave", "opera", "ucmobile", "microsoft.emmx"],
    },
    "map_navigation": {
        "label": "地图导航类应用",
        "keywords": ["地图", "导航", "路线", "map", "navigation", "高德", "百度地图", "腾讯地图"],
        "package_keywords": ["map", "maps", "navigation", "autonavi", "baidu.map", "tencent.map", "minimap"],
    },
    "messaging": {
        "label": "聊天通讯类应用",
        "keywords": ["聊天", "通讯", "消息", "微信", "qq", "telegram", "whatsapp", "飞书", "企业微信"],
        "package_keywords": ["wechat", "tencent.mm", "mobileqq", "telegram", "whatsapp", "lark", "wecom"],
    },
    "email": {
        "label": "邮箱类应用",
        "keywords": ["邮箱", "邮件", "email", "mail", "gmail", "outlook", "网易邮箱", "qq邮箱"],
        "package_keywords": ["mail", "email", "gmail", "outlook"],
    },
    "office": {
        "label": "办公文档类应用",
        "keywords": ["办公", "文档", "表格", "word", "excel", "wps", "office", "飞书", "钉钉", "腾讯文档"],
        "package_keywords": ["office", "wps", "docs", "document", "lark", "dingtalk"],
    },
    "calendar": {
        "label": "日历日程类应用",
        "keywords": ["日历", "日程", "calendar", "提醒", "待办"],
        "package_keywords": ["calendar", "schedule", "todo", "reminder"],
    },
    "notes": {
        "label": "笔记备忘类应用",
        "keywords": ["笔记", "备忘", "便签", "记事", "note", "notes", "memo"],
        "package_keywords": ["note", "notes", "memo", "notepad"],
    },
    "video": {
        "label": "视频播放类应用",
        "keywords": ["视频", "播放器", "看剧", "video", "player", "bilibili", "优酷", "腾讯视频", "爱奇艺"],
        "package_keywords": ["video", "player", "bilibili", "youku", "qiyi", "iqiyi"],
    },
    "music": {
        "label": "音乐类应用",
        "keywords": ["音乐", "听歌", "music", "网易云", "qq音乐", "酷狗", "酷我"],
        "package_keywords": ["music", "netease.cloudmusic", "kugou", "kuwo"],
    },
    "payment": {
        "label": "支付类应用",
        "keywords": ["支付", "付款", "钱包", "支付宝", "微信支付", "pay", "wallet"],
        "package_keywords": ["pay", "wallet", "alipay", "tencent.mm"],
    },
    "shopping": {
        "label": "购物类应用",
        "keywords": ["购物", "电商", "淘宝", "京东", "拼多多", "shop", "shopping"],
        "package_keywords": ["taobao", "jingdong", "jd", "pinduoduo", "shopping", "mall"],
    },
    "delivery": {
        "label": "外卖配送类应用",
        "keywords": ["外卖", "配送", "点餐", "美团", "饿了么", "delivery"],
        "package_keywords": ["meituan", "eleme", "delivery", "takeout"],
    },
    "travel": {
        "label": "出行旅行类应用",
        "keywords": ["出行", "旅行", "打车", "订票", "酒店", "trip", "travel", "滴滴", "携程", "飞猪"],
        "package_keywords": ["travel", "trip", "didi", "ctrip", "fliggy", "hotel"],
    },
    "cloud_drive": {
        "label": "网盘云盘类应用",
        "keywords": ["网盘", "云盘", "云存储", "cloud", "drive", "百度网盘", "阿里云盘"],
        "package_keywords": ["cloud", "drive", "netdisk", "pan.baidu", "aliyun"],
    },
    "file_manager": {
        "label": "文件管理类应用",
        "keywords": ["文件", "文件管理", "管理器", "file", "manager"],
        "package_keywords": ["file", "manager", "explorer"],
    },
    "camera_gallery": {
        "label": "相机相册类应用",
        "keywords": ["相机", "拍照", "相册", "图库", "照片", "camera", "gallery", "photo"],
        "package_keywords": ["camera", "gallery", "photo", "album"],
    },
}

KNOWN_EXACT_APPS: dict[str, tuple[str, ...]] = {
    "微信": ("微信", "wechat", "tencent.mm"),
    "高德地图": ("高德地图", "autonavi", "minimap"),
    "chrome": ("chrome", "谷歌浏览器", "com.android.chrome"),
    "qq": ("qq", "mobileqq"),
    "飞书": ("飞书", "lark", "feishu"),
    "企业微信": ("企业微信", "wecom", "wxwork"),
}

CUSTOM_KEYWORD_HINTS: dict[str, tuple[str, ...]] = {
    "会议": ("会议", "视频会议", "meeting", "conference", "zoom", "腾讯会议", "飞书会议"),
    "会议软件": ("会议", "视频会议", "meeting", "conference", "zoom", "腾讯会议", "飞书会议"),
}

INVENTORY_MARKERS = (
    "应用列表",
    "已安装应用",
    "手机应用",
    "app列表",
    "installed apps",
    "应用清单",
)
QUERY_MARKERS = ("是否有", "有没有", "查找", "搜索", "找一下", "哪些", "某类应用", "相关应用", "类应用")
PHONE_QUERY_MARKERS = ("手机", "手机里", "手机上", "应用")
QUESTION_MARKERS = ("有没有", "有哪些", "有", "找", "看看")


@dataclass(frozen=True)
class AppInventoryIntent:
    should_handle: bool
    query_text: str
    query_kind: AppQueryKind
    category_key: str | None
    category_label: str
    app_type: AppType = "all"
    keywords: list[str] = field(default_factory=list)
    package_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MatchedApp:
    app_label: str
    package_name: str
    matched_keyword: str | None
    score: int


class AppInventoryQueryMiddleware(AgentMiddleware[MobileAgentState, None, Any]):
    state_schema = MobileAgentState

    def __init__(self, system_gateway: SystemToolGateway) -> None:
        self.runner = AppInventoryQueryRunner(system_gateway)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        intent = _intent_from_request(request.messages)
        if not intent.should_handle:
            return handler(request)
        result = asyncio.run(self.runner.run(intent, _request_device_id(request)))
        return _model_response(result)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        intent = _intent_from_request(request.messages)
        if not intent.should_handle:
            return await handler(request)
        result = await self.runner.run(intent, _request_device_id(request))
        return _model_response(result)


@dataclass(frozen=True)
class AppInventoryQueryRunner:
    system_gateway: SystemToolGateway

    async def run(self, intent: AppInventoryIntent, device_id: str | None = None) -> JsonObject:
        emit_task_complexity(
            complexity="simple",
            track_steps=True,
            reason=APP_INVENTORY_PHASE,
            message="识别任务类型：应用列表读取与目标检索",
        )

        self._emit_progress(
            step=1,
            status="running",
            step_title="分析检索目标",
            message="正在分析需要检索的应用目标。",
            tool_name="app_inventory_intent",
            completed_count=0,
        )
        self._emit_progress(
            step=1,
            status="completed",
            step_title="分析检索目标",
            message=f"已识别为检索：{intent.category_label}。",
            tool_name="app_inventory_intent",
            completed_count=1,
        )

        self._emit_progress(
            step=2,
            status="running",
            step_title="读取手机应用列表",
            message="正在读取手机上的所有已安装应用。",
            tool_name="list_apps",
            completed_count=1,
        )
        apps_result = await self._read_apps(intent, device_id)
        if "error" in apps_result:
            final_message = str(apps_result["error"])
            self._emit_progress(
                step=2,
                status="failed",
                step_title="读取手机应用列表",
                message=final_message,
                tool_name="list_apps",
                completed_count=1,
                error=final_message,
            )
            return self._result(intent, {}, [], final_message)

        apps = cast(dict[str, str], apps_result["apps"])
        total_apps = len(apps)
        self._emit_progress(
            step=2,
            status="completed",
            step_title="读取手机应用列表",
            message=f"已完成应用列表读取，共读取 {total_apps} 个应用。",
            tool_name="list_apps",
            completed_count=2,
        )

        self._emit_progress(
            step=3,
            status="running",
            step_title="过滤检索结果",
            message=f"正在根据“{intent.category_label}”过滤检索结果。",
            tool_name="app_inventory_filter",
            completed_count=2,
        )
        matches = filter_apps(apps, intent)
        matched_count = len(matches)
        self._emit_progress(
            step=3,
            status="completed",
            step_title="过滤检索结果",
            message=f"已找到 {matched_count} 个{intent.category_label}。",
            tool_name="app_inventory_filter",
            completed_count=3,
        )

        final_message = format_app_inventory_result(intent, matches)
        self._emit_progress(
            step=4,
            status="completed",
            step_title="整理检索结果",
            message=final_message,
            tool_name="finish",
            completed_count=4,
        )
        return self._result(intent, apps, matches, final_message)

    async def _read_apps(
        self,
        intent: AppInventoryIntent,
        device_id: str | None,
    ) -> JsonObject:
        try:
            client = self.system_gateway.get_default_client(device_id)
            result = await client.send_request("listApps", {"type": intent.app_type})
        except SystemGatewayError as exc:
            return {"error": _system_gateway_error_message(exc)}
        except Exception as exc:
            return {"error": f"应用列表读取失败：{exc}"}
        if not isinstance(result, Mapping):
            return {"error": "应用列表读取失败：系统工具返回的应用列表格式不正确。"}
        apps: dict[str, str] = {}
        for package_name, app_label in result.items():
            if isinstance(package_name, str) and isinstance(app_label, str):
                apps[package_name] = app_label
        return {"apps": apps}

    def _emit_progress(
        self,
        *,
        step: int,
        status: str,
        step_title: str,
        message: str,
        tool_name: str,
        completed_count: int,
        error: str | None = None,
    ) -> None:
        emit_task_progress(
            label=tool_name,
            status=cast(Any, status),
            phase=APP_INVENTORY_PHASE,
            task_title=APP_INVENTORY_TASK_TITLE,
            step_title=step_title,
            message=message,
            tool_name=tool_name,
            progress_key=f"app-inventory-query-{step}",
            current_step=step,
            total_steps=TOTAL_STEPS,
            completed_steps=_completed_steps(completed_count),
            error=error,
        )

    def _result(
        self,
        intent: AppInventoryIntent,
        apps: Mapping[str, str],
        matches: Sequence[MatchedApp],
        final_message: str,
    ) -> JsonObject:
        return {
            "queryText": intent.query_text,
            "queryKind": intent.query_kind,
            "categoryKey": intent.category_key,
            "categoryLabel": intent.category_label,
            "totalApps": len(apps),
            "matchedCount": len(matches),
            "matches": [_matched_app_json(match) for match in matches],
            "finalMessage": final_message,
        }


def is_app_inventory_query_request(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    has_inventory_marker = any(marker.lower() in normalized for marker in INVENTORY_MARKERS)
    has_query_marker = any(marker.lower() in normalized for marker in QUERY_MARKERS)
    if has_inventory_marker and has_query_marker:
        return True
    has_phone_marker = any(marker in normalized for marker in PHONE_QUERY_MARKERS)
    has_question_marker = any(marker in normalized for marker in QUESTION_MARKERS)
    has_known_target = _detect_category(normalized) is not None or _detect_exact_app(normalized) is not None
    return has_phone_marker and has_question_marker and has_known_target


def parse_app_inventory_intent(text: str) -> AppInventoryIntent:
    normalized = _normalize(text)
    should_handle = is_app_inventory_query_request(text)
    category_key = _detect_category(normalized)
    exact_app = _detect_exact_app(normalized)
    if exact_app is not None:
        keywords = list(KNOWN_EXACT_APPS[exact_app])
        return AppInventoryIntent(
            should_handle=should_handle,
            query_text=exact_app,
            query_kind="exact_app",
            category_key=None,
            category_label=exact_app,
            keywords=keywords,
            package_keywords=[keyword for keyword in keywords if re.search(r"[a-z.]", keyword.lower())],
        )
    if category_key is not None:
        entry = APP_CATEGORY_REGISTRY[category_key]
        return AppInventoryIntent(
            should_handle=should_handle,
            query_text=str(entry["label"]).removesuffix("类应用"),
            query_kind="category",
            category_key=category_key,
            category_label=str(entry["label"]),
            keywords=list(cast(Sequence[str], entry["keywords"])),
            package_keywords=list(cast(Sequence[str], entry["package_keywords"])),
        )
    query_text = _extract_custom_query_text(text)
    custom_keywords = _custom_keywords(query_text)
    return AppInventoryIntent(
        should_handle=should_handle,
        query_text=query_text,
        query_kind="custom",
        category_key="custom",
        category_label=query_text,
        keywords=custom_keywords,
        package_keywords=[keyword for keyword in custom_keywords if re.search(r"[a-z.]", keyword.lower())],
    )


def filter_apps(apps: dict[str, str], intent: AppInventoryIntent) -> list[MatchedApp]:
    matches: dict[str, MatchedApp] = {}
    for package_name, app_label in apps.items():
        match = _score_app(package_name, app_label, intent)
        if match is None or match.score < 50:
            continue
        existing = matches.get(package_name)
        if existing is None or match.score > existing.score:
            matches[package_name] = match
    return sorted(matches.values(), key=lambda item: (-item.score, item.app_label, item.package_name))


def format_app_inventory_result(
    intent: AppInventoryIntent,
    matches: Sequence[MatchedApp],
) -> str:
    if not matches:
        return f"已读取当前手机应用列表。未检测到明确的{intent.category_label}。"
    lines = [
        f"已读取当前手机应用列表。共找到 {len(matches)} 个{intent.category_label}：",
        "",
    ]
    lines.extend(
        f"{index}. {match.app_label}（{match.package_name}）"
        for index, match in enumerate(matches, start=1)
    )
    return "\n".join(lines)


def _score_app(package_name: str, app_label: str, intent: AppInventoryIntent) -> MatchedApp | None:
    label_lower = app_label.lower()
    package_lower = package_name.lower()
    query_lower = intent.query_text.lower()
    best_score = 0
    best_keyword: str | None = None

    def update(score: int, keyword: str | None) -> None:
        nonlocal best_score, best_keyword
        if score > best_score:
            best_score = score
            best_keyword = keyword

    if intent.query_kind == "exact_app":
        if label_lower == query_lower:
            update(100, intent.query_text)
        for keyword in [intent.query_text, *intent.keywords]:
            keyword_lower = keyword.lower()
            if not keyword_lower:
                continue
            if label_lower == keyword_lower:
                update(100, keyword)
            if keyword_lower in label_lower:
                update(80, keyword)
            if keyword_lower in package_lower:
                update(60, keyword)
        if difflib.SequenceMatcher(None, label_lower, query_lower).ratio() >= 0.65:
            update(50, intent.query_text)
    else:
        for keyword in intent.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower and keyword_lower in label_lower:
                update(80, keyword)
        for keyword in intent.package_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower and keyword_lower in package_lower:
                update(70, keyword)
        if query_lower and query_lower in label_lower:
            update(60, intent.query_text)
        if query_lower and query_lower in package_lower:
            update(50, intent.query_text)
        for keyword in intent.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower and difflib.SequenceMatcher(None, label_lower, keyword_lower).ratio() >= 0.62:
                update(50, keyword)

    if best_score < 50:
        return None
    return MatchedApp(
        app_label=app_label,
        package_name=package_name,
        matched_keyword=best_keyword,
        score=best_score,
    )


def _detect_category(normalized: str) -> str | None:
    for key, entry in APP_CATEGORY_REGISTRY.items():
        keywords = cast(Sequence[str], entry["keywords"])
        if any(keyword.lower() in normalized for keyword in keywords):
            return key
    if "能开网页" in normalized or "看网页" in normalized:
        return "browser"
    return None


def _detect_exact_app(normalized: str) -> str | None:
    for app_name, aliases in KNOWN_EXACT_APPS.items():
        if any(alias.lower() in normalized for alias in aliases):
            return app_name
    return None


def _extract_custom_query_text(text: str) -> str:
    cleaned = re.sub(r"[。！？?!.，,]", " ", text).strip()
    patterns = (
        r"找一下(?P<target>.+?)(相关的)?应用",
        r"搜索(?P<target>.+?)(相关的)?应用",
        r"查找(?P<target>.+?)(相关的)?应用",
        r"有没有(?P<target>.+?)(类应用|应用|软件)?$",
        r"有哪些(?P<target>.+?)(类应用|应用|软件)?$",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return _clean_query_text(match.group("target"))
    return _clean_query_text(cleaned) or "相关应用"


def _clean_query_text(value: str) -> str:
    cleaned = re.sub(r"^(读取|当前|手机|手机上|手机里|帮我|看看|我)?", "", value.strip())
    cleaned = re.sub(r"(相关的)?(类应用|应用|软件)$", "", cleaned.strip())
    return cleaned.strip() or value.strip()


def _custom_keywords(query_text: str) -> list[str]:
    keywords: list[str] = [query_text]
    normalized_query = _normalize(query_text)
    for trigger, additions in CUSTOM_KEYWORD_HINTS.items():
        if trigger in normalized_query:
            keywords.extend(additions)
    return list(dict.fromkeys(keyword for keyword in keywords if keyword))


def _intent_from_request(messages: Sequence[BaseMessage]) -> AppInventoryIntent:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return parse_app_inventory_intent(_message_content_to_text(message.content))
    return AppInventoryIntent(
        should_handle=False,
        query_text="",
        query_kind="custom",
        category_key=None,
        category_label="相关应用",
    )


def _request_device_id(request: ModelRequest[Any]) -> str | None:
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None)
    if isinstance(config, Mapping):
        device_id = (
            device_id_from_mapping(config.get("configurable"))
            or device_id_from_mapping(config.get("metadata"))
        )
        if device_id is not None:
            return device_id
    state = getattr(request, "state", None)
    return device_id_from_mapping(state)


def _model_response(result: JsonObject) -> ModelResponse[Any]:
    return ModelResponse(
        result=[
            AIMessage(
                content=str(result["finalMessage"]),
                additional_kwargs={"app_inventory_query": result},
            )
        ]
    )


def _system_gateway_error_message(exc: SystemGatewayError) -> str:
    message = str(exc)
    if "Multiple system tool clients are connected" in message:
        return "应用列表读取失败：检测到多个设备连接，但当前请求未绑定设备 ID。"
    if "No connected system tool client is available" in message:
        return "应用列表读取失败：系统工具客户端未连接。"
    return f"应用列表读取失败：{message}"


def _completed_steps(count: int) -> list[JsonObject]:
    definitions = (
        ("分析检索目标", "app_inventory_intent"),
        ("读取手机应用列表", "list_apps"),
        ("过滤检索结果", "app_inventory_filter"),
        ("整理检索结果", "finish"),
    )
    return [
        {
            "index": index,
            "name": name,
            "toolName": tool_name,
            "status": "completed",
        }
        for index, (name, tool_name) in enumerate(definitions, start=1)
        if index <= count
    ]


def _matched_app_json(match: MatchedApp) -> JsonObject:
    return cast(JsonObject, to_json_value({
        "appLabel": match.app_label,
        "packageName": match.package_name,
        "matchedKeyword": match.matched_keyword,
        "score": match.score,
    }))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


__all__ = [
    "APP_CATEGORY_REGISTRY",
    "AppInventoryIntent",
    "AppInventoryQueryMiddleware",
    "AppInventoryQueryRunner",
    "MatchedApp",
    "filter_apps",
    "format_app_inventory_result",
    "is_app_inventory_query_request",
    "parse_app_inventory_intent",
]
