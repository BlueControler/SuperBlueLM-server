# OpenRouter Run Guide

当前代码已经支持优先使用 OpenRouter。

## 环境变量

PowerShell:

```powershell
$env:OPENROUTER_API_KEY="你的-openrouter-key"
$env:OPENROUTER_MODEL="openai/gpt-5.1"
```

如果设置了 `OPENROUTER_API_KEY`，`custom_deep_agent.py` 会优先走：

- base_url: `https://openrouter.ai/api/v1`
- model: `OPENROUTER_MODEL`

如果没有设置 `OPENROUTER_API_KEY`，则会退回到：

- `openai:gpt-5.4`

## 运行

1. 启动 agent 服务端

```bash
python agent_server.py
```

2. 另开一个终端启动 mock 手机端

```bash
python mock_portal_client.py
```

3. 在 `agent_server.py` 的命令行输入自然语言任务

例如：

```text
打开设置，然后点击一个按钮
```

## 说明

- 没有真机时，依然可以用 `mock_portal_client.py` 验证真实模型是否会调用 websocket tools
- websocket tools、remove_old_images 和协议逻辑不需要因为 OpenRouter 再做额外改动
