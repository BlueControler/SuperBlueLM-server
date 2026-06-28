from mobile_agent.prompt_assets import PHONE_SUBAGENT_SYSTEM_PROMPT, SYSTEM_PROMPT


def test_main_prompt_assigns_planning_execution_and_result_review_to_main_agent() -> None:
    assert "维护执行计划" in SYSTEM_PROMPT
    assert "直接调用手机工具" in SYSTEM_PROMPT
    assert "读取工具返回结果" in SYSTEM_PROMPT
    assert "继续修正" in SYSTEM_PROMPT
    assert "目标完成后再给用户最终答复" in SYSTEM_PROMPT
    assert "必须调用 execute_phone_todo" not in SYSTEM_PROMPT
    assert "子 agent" not in SYSTEM_PROMPT


def test_legacy_phone_subagent_prompt_is_not_main_agent_contract() -> None:
    assert "只执行主 agent 下发的一条明确 TODO" in PHONE_SUBAGENT_SYSTEM_PROMPT
    assert "顺序执行完成当前 TODO 所需的少量动作" in PHONE_SUBAGENT_SYSTEM_PROMPT
