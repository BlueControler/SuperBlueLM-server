from mobile_agent.prompt_assets import PHONE_SUBAGENT_SYSTEM_PROMPT, SYSTEM_PROMPT


def test_main_prompt_assigns_planning_validation_and_correction_to_main_agent() -> None:
    assert "维护执行计划" in SYSTEM_PROMPT
    assert "决定下一条 TODO" in SYSTEM_PROMPT
    assert "验收结果" in SYSTEM_PROMPT
    assert "纠正偏差" in SYSTEM_PROMPT
    assert "追加一条纠正 TODO" in SYSTEM_PROMPT


def test_phone_subagent_prompt_allows_bounded_sequential_execution() -> None:
    assert "只执行主 agent 下发的一条明确 TODO" in PHONE_SUBAGENT_SYSTEM_PROMPT
    assert "每次操作前使用最新截图和 UI 树" in PHONE_SUBAGENT_SYSTEM_PROMPT
    assert "顺序执行完成当前 TODO 所需的少量动作" in PHONE_SUBAGENT_SYSTEM_PROMPT
    assert "禁止并行执行手机操作" in PHONE_SUBAGENT_SYSTEM_PROMPT
    assert "工具预算耗尽或同一动作重复失败时" in PHONE_SUBAGENT_SYSTEM_PROMPT
    assert "停止并把原因交还主 agent" in PHONE_SUBAGENT_SYSTEM_PROMPT
