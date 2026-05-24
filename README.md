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

1. 手机操作客户端连接 `/adb`，首条消息发送 `connect`。
2. 系统工具客户端连接 `/system`，提供应用列表、日程、提醒、定位等 API。
3. Agent 通过工具向两个 WebSocket client 下发请求。
4. 手机端或系统工具端返回结果，服务端更新状态并继续下一步。

## 部署

从仓库根目录执行一条命令即可完成 llama.cpp 和外部业务工具部署：

```bash
python -m setup deploy
```

这条命令默认执行 `full` profile，流程只有两步：

1. 安装 llama.cpp 并创建本地模型目录。
2. 安装飞书、企业微信 CLI，并检查高德 MCP 等外部业务工具。

可选 profile 仍然保留，方便只部署一部分：

- `local`: 安装 llama.cpp 并创建本地模型目录（需要手动放入模型文件）。
- `external`: 安装飞书、企业微信 CLI（不做登录授权）。
- `full`: `local` + `external`，先完成本地模型，再安装外部工具。

例如只部署本地模型：

```bash
python -m setup deploy --profile local
```

如果只想打印完整部署步骤，不执行安装：

```bash
python -m setup deploy --dry-run
```

然后到Hugging Face或镜像站下载 GGUF 模型文件，放到 `.local/models/` 目录下即可。

### 离线本地模型

安装 llama.cpp 预编译包并创建 models 目录（不再自动下载模型）：

```bash
python -m setup llama:all
```

脚本会自动识别 Windows x64、Linux x64、Android arm64，也可以显式指定：

```bash
python -m setup llama:all --target linux-x64
```

默认目录在仓库内 `.local/`。模型文件请手动放到 `.local/models/`，例如保持默认文件名就放在 `.local/models/gemma-4-E2B-it-Q8_0.gguf`；或自行指定路径并设置 `LLAMA_CPP_MODEL_PATH` 指向你的 GGUF 文件。

也可以用环境变量覆盖：

- `LLAMA_CPP_SERVER_BINARY`: `llama-server` 可执行文件路径
- `LLAMA_CPP_MODEL_PATH`: GGUF 模型文件路径
- `LLAMA_CPP_HOST` / `LLAMA_CPP_PORT`: 本地 llama.cpp server 地址，默认 `127.0.0.1:8080`
- `LLAMA_CPP_MODEL_NAME`: OpenAI-compatible model 名称，默认 `gemma-4-E2B-it`

统一 setup 入口是 `python -m setup`。

本地模型模式会额外注入一份更保守的系统提示词，只允许简单任务、零次或一次工具调用；多步骤、高风险或不确定任务会停止并要求用户确认或交还给更强模型。

### 外部业务工具

飞书、企业微信和高德 MCP 工具统一通过 setup 入口安装或检查：

```bash
python -m setup external:check
python -m setup external:all
```

高德 MCP 使用官方 Streamable HTTP 地址 `https://mcp.amap.com/mcp?key=...`。需要设置 `AMAP_MAPS_API_KEY`；如需切换网关或代理，可用 `AMAP_MCP_HTTP_URL` 覆盖基础 URL。

切换网络状态：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:2024/network/status -ContentType "application/json" -Body '{"connected": false}'
```

当 `connected=false` 时，服务会启动本地 `llama-server` 并让 agent graph 切到本地模型；当 `connected=true` 时，会切回云端并停止本地进程。

## 运行

```bash
langgraph dev
```

## Nginx 端口转发

如需把 `langgraph dev` 的默认端口 `127.0.0.1:2024` 暴露到其他地址/端口，可在本机 Nginx 配置中加入一个反向代理服务。关键点是保留 WebSocket 和 SSE 的相关设置。配置见 `nginx.conf`。