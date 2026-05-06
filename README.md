# IWebsocket-server

## 项目作用

这个项目是一个 Android 手机远程操作服务端。

它通过 WebSocket 与手机侧工具通信，并在服务端把手机能力包装成工具（observe、tap、swipe、type、keyevent 等），供 Deep Agent 进行任务规划和自动执行。

当前代码已按单设备模型实现：

- 手机操作 WebSocket 路径固定为 `/adb`
- 系统工具 WebSocket 路径固定为 `/system`
- 同一时刻只允许 1 台设备连接
- Agent 每轮可基于最新截图和 UI 树决策

## 核心流程

1. 使用 `langgraph dev` 启动 LangGraph API Server。
2. 手机操作客户端连接 `ws://host:port/adb`，首条消息发送 `connect`。
3. 系统工具客户端连接 `ws://host:port/system`，提供应用列表、日程、提醒、定位等 API。
4. LangGraph API 调用 `agent` graph，Agent 通过工具向两个 WebSocket client 下发请求。
5. 手机端或系统工具端返回结果，服务端更新状态并继续下一步。

## 开发环境要求

- Python `>= 3.14`
- 建议在虚拟环境中开发

## 安装依赖（开发模式）

### 方式一：使用 uv

```bash
uv sync
```

如果你希望包含开发依赖（ruff、mypy、pytest、langgraph-cli 等），使用：

```bash
uv sync --group dev
```

### 方式二：使用 pip

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

安装项目（可编辑模式）以及开发依赖：

```bash
pip install -e .
pip install mypy ruff pytest "langgraph-cli[inmem]" anyio
```

## 正式运行：LangGraph API

项目的正式入口是 `langgraph dev`。它会读取 `langgraph.json`，同时启动：

- LangGraph API / Studio 调试服务
- `agent` graph
- 自定义 HTTP app 中的 `/adb` 和 `/system` WebSocket 路由
- `/adb/status` 和 `/system/status` 状态检查接口

```bash
langgraph dev --port 2024
```

启动后：

- LangGraph API: `http://127.0.0.1:2024`
- Studio UI: 终端输出的 `https://smith.langchain.com/studio/?baseUrl=...`
- 手机操作客户端连接：`ws://127.0.0.1:2024/adb`
- 系统工具客户端连接：`ws://127.0.0.1:2024/system`
- 手机连接状态：`http://127.0.0.1:2024/adb/status`
- 系统工具连接状态：`http://127.0.0.1:2024/system/status`
- 网络/模型路由状态：`http://127.0.0.1:2024/network/status`

LangSmith tracing 使用 `.env` 中的 `LANGSMITH_TRACING=true`、`LANGSMITH_PROJECT` 和 `LANGSMITH_API_KEY`。

## 离线本地模型

安装 llama.cpp 预编译包并下载默认 Gemma 4 GGUF 模型：

```bash
python -m entrypoints.llama_cpp_setup all
```

脚本会自动识别 Windows x64、Linux x64、Android arm64，也可以显式指定：

```bash
python -m entrypoints.llama_cpp_setup all --target linux-x64
```

如果 Hugging Face 需要授权，先设置 `HF_TOKEN` 或 `HUGGINGFACE_TOKEN`。默认文件会放到仓库内 `.local/`，也可以用环境变量覆盖：

- `LLAMA_CPP_SERVER_BINARY`: `llama-server` 可执行文件路径
- `LLAMA_CPP_MODEL_PATH`: GGUF 模型文件路径
- `LLAMA_CPP_HOST` / `LLAMA_CPP_PORT`: 本地 llama.cpp server 地址，默认 `127.0.0.1:8080`
- `LLAMA_CPP_MODEL_NAME`: OpenAI-compatible model 名称，默认 `gemma-4-E2B-it`

切换网络状态：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:2024/network/status -ContentType "application/json" -Body '{"connected": false}'
```

当 `connected=false` 时，服务会启动本地 `llama-server` 并让 agent graph 切到本地模型；当 `connected=true` 时，会切回云端并停止本地进程。

如果看到 Windows 的 `WinError 10048`，表示端口已经被另一个进程占用。可以换端口：

```bash
langgraph dev --port 2025
```

或查看占用进程：

```powershell
Get-NetTCPConnection -LocalPort 2024 | Select-Object LocalAddress,LocalPort,State,OwningProcess
```

## 调试 entrypoint

### 1) `entrypoints.main`

手工调试 WebSocket 通道用的控制台服务。

- 监听 `/adb`
- 接收客户端消息并打印
- 支持在终端输入任意文本，广播到已连接客户端

启动：

```bash
python -m entrypoints.main --host 127.0.0.1 --port 8765
```

### 2) `entrypoints.agent_server`

旧的终端 Agent 控制入口，适合不用 LangGraph API 时手工调试。

- 启动 `DeviceGateway` 并接收手机连接
- 构建 Deep Agent 与手机工具集
- 在终端输入自然语言任务，Agent 自动调用工具执行

启动：

```bash
python -m entrypoints.agent_server --host 127.0.0.1 --port 8765
```

### 3) `entrypoints.mock_portal_client`

本地模拟手机端客户端，便于联调服务端协议和工具调用链路。

- 连到 `/adb`
- 发送 `connect` 和 `ping`
- 对服务端请求返回 mock 的 `actionResult`

启动：

```bash
python -m entrypoints.mock_portal_client
```

## 快速联调示例

1. 终端 A 启动 LangGraph API：

```bash
langgraph dev --port 8765
```

2. 终端 B 启动模拟手机端：

```bash
python -m entrypoints.mock_portal_client
```

3. 打开 Studio UI，选择 `agent`，在同一个 thread 中输入任务，例如：

```text
打开系统设置，然后返回桌面
```

