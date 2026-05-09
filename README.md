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

## 一键部署

从仓库根目录执行一条命令即可完成完整部署：

```bash
python -m entrypoints.deploy
```

这条命令默认执行 `full` profile 并启动 LangGraph 服务，流程包括：

1. 检查 Python 版本是否满足 `>=3.14`。
2. 如果 `.env` 不存在，从 `.env.example` 创建。
3. 检查启动端口是否可用。
4. 安装 Python 依赖。
5. 安装 llama.cpp 并下载默认本地模型。
6. 安装并初始化飞书、企业微信外部工具。
7. 检查统一 setup 状态。
8. 启动 LangGraph 服务。
9. 请求 `/network/status` 做部署后健康检查。

Windows PowerShell 也可以使用：

```powershell
.\scripts\deploy.ps1
```

Linux/macOS 也可以使用：

```bash
sh scripts/deploy.sh
```

可选 profile 仍然保留，方便只部署一部分：

- `core`: 安装 Python 依赖并检查本地模型、外部工具状态；可选组件缺失只会给出 warning。
- `local`: 在 `core` 基础上安装 llama.cpp 并下载默认本地模型。
- `full`: 在 `local` 基础上安装并初始化飞书、企业微信外部工具；该模式会触发 CLI 登录或初始化流程。

飞书/企微 CLI 的授权是“首次部署需要人工处理，后续复用登录态”的模式：第一次执行 `full` 时，如果 CLI 要求扫码、浏览器登录或填写配置，需要按提示完成；完成后登录态通常保存在本机 CLI 配置目录里。之后在同一台机器、同一用户下再次执行一键部署，会优先复用已有登录态，一般不会重复登录。只有登录态过期、被清理、换机器或换系统用户时，才需要重新授权。

例如只部署本地模型并启动：

```bash
python -m entrypoints.deploy --profile local --start --port 2024
```

如果只想打印完整部署步骤，不执行下载、登录或启动：

```bash
python -m entrypoints.deploy --dry-run
```

如果本机已装好依赖，可跳过依赖安装；如果只是临时诊断，也可以跳过 Python 版本保护：

```bash
python -m entrypoints.deploy --profile core --no-start --no-install-deps --allow-unsupported-python
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
python -m entrypoints.setup llama:all
```

脚本会自动识别 Windows x64、Linux x64、Android arm64，也可以显式指定：

```bash
python -m entrypoints.setup llama:all --target linux-x64
```

如果 Hugging Face 需要授权，先设置 `HF_TOKEN` 或 `HUGGINGFACE_TOKEN`。默认文件会放到仓库内 `.local/`，也可以用环境变量覆盖：

- `LLAMA_CPP_SERVER_BINARY`: `llama-server` 可执行文件路径
- `LLAMA_CPP_MODEL_PATH`: GGUF 模型文件路径
- `LLAMA_CPP_HOST` / `LLAMA_CPP_PORT`: 本地 llama.cpp server 地址，默认 `127.0.0.1:8080`
- `LLAMA_CPP_MODEL_NAME`: OpenAI-compatible model 名称，默认 `gemma-4-E2B-it`

旧入口 `python -m entrypoints.llama_cpp_setup all` 仍可用，会转发到统一 setup。

本地模型模式会额外注入一份更保守的系统提示词，只允许简单任务、零次或一次工具调用；多步骤、高风险或不确定任务会停止并要求用户确认或交还给更强模型。

## 外部业务工具

飞书、企业微信和高德 MCP 工具统一通过 setup 入口安装或检查：

```bash
python -m entrypoints.setup external:check
python -m entrypoints.setup external:all
```

旧入口 `python -m entrypoints.external_tools_setup all` 仍可用，会转发到统一 setup。

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

