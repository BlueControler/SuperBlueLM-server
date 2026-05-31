from mobile_agent.prompt_assets import PHONE_SUBAGENT_SYSTEM_PROMPT, SYSTEM_PROMPT


def test_main_prompt_assigns_planning_validation_and_correction_to_main_agent() -> None:
    assert "维护执行计划" in SYSTEM_PROMPT
    assert "决定下一条 TODO" in SYSTEM_PROMPT
    assert "验收结果" in SYSTEM_PROMPT
    assert "纠正偏差" in SYSTEM_PROMPT
    assert "追加一条纠正 TODO" in SYSTEM_PROMPT


def test_phone_subagent_prompt_limits_child_to_one_clear_execution_chain() -> None:
    assert "只执行主 agent 下发的一条明确 TODO" in PHONE_SUBAGENT_SYSTEM_PROMPT
    assert "每次操作前使用最新截图和 UI 树" in PHONE_SUBAGENT_SYSTEM_PROMPT
    assert "只有任务明确允许短链时" in PHONE_SUBAGENT_SYSTEM_PROMPT
    assert "停止并把原因交还主 agent" in PHONE_SUBAGENT_SYSTEM_PROMPT
