from mobile_agent.prompt_assets import SYSTEM_PROMPT


def test_main_prompt_assigns_planning_execution_and_result_review_to_main_agent() -> None:
    assert "维护执行计划" in SYSTEM_PROMPT
    assert "直接调用手机工具" in SYSTEM_PROMPT
    assert "读取工具返回结果" in SYSTEM_PROMPT
    assert "继续修正" in SYSTEM_PROMPT
    assert "目标完成后再给用户最终答复" in SYSTEM_PROMPT
    assert "必须调用 execute_phone_todo" not in SYSTEM_PROMPT
    assert "子 agent" not in SYSTEM_PROMPT
