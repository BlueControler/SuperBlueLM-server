from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware.types import AgentState, StateT, before_model
from langgraph.runtime import Runtime


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@before_model
def remove_old_images(
    state: AgentState[StateT],
    runtime: Runtime,
) -> dict[str, Any] | None:
    filtered_messages = state["messages"][:-1]
    # 对message进行处理，移除其中的旧图片，只保留最新一张
    # 这里写个假业务逻辑，实际中你需要根据message的结构来实现这个功能

    return {
        "messages": filtered_messages,
    }


agent = create_deep_agent(
    model="openai:gpt-5.4",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    middleware=[remove_old_images],
)

# Run the agent
agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
