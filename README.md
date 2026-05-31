# IWebsocket-server

## 项目作用

这个项目是一个 Android 手机远程操作服务端。

它通过 WebSocket 与手机侧工具通信，并在服务端把手机能力包装成工具（observe、tap、swipe、type、keyevent 等）。云端模式下，主 agent 负责规划、验收和纠错，手机子 agent 负责把一条明确 TODO 转换成受限手机工具调用。

当前代码已按单设备模型实现：

- 手机操作 WebSocket 路径固定为 `/adb`
- 系统工具 WebSocket 路径固定为 `/system`
- 同一时刻只允许 1 台设备连接
- Agent 每轮可基于最新截图和 UI 树决策

## 核心流程

1. 手机操作客户端连接 `/adb`，首条消息发送 `connect`。
2. 系统工具客户端连接 `/system`，提供应用列表、日程、提醒、定位等 API。
3. 主 agent 直接调用系统工具和外部业务工具；手机 UI 操作通过 `execute_phone_todo` 委派给手机子 agent。
4. 手机子 agent 仅持有手机工具，并在受限调用预算内执行当前 TODO。
5. 手机端或系统工具端返回结果，服务端更新状态并继续下一步。

## 多模型 Agent 架构

网络可用时：

```text
主 agent -> execute_phone_todo -> 手机子 agent -> observe/tap/type/swipe 等手机工具
主 agent -> 系统工具和外部业务工具
```

- 主 agent 使用云端强模型，负责维护计划、创建或修正 TODO、检查执行结果并决定是否结束。
- 手机子 agent 使用独立可配置模型，只执行一条明确手机 TODO。
- `allow_short_chain=false` 时，子 agent 默认只执行一个手机工具调用。
- `allow_short_chain=true` 时，子 agent 可以执行少量确定性连续动作，但仍受调用预算限制。
- 手机子 agent 不持有系统、飞书、企业微信、地图或天气工具。
- 云端主 agent 看不到底层手机工具，不能绕过 `execute_phone_todo`。

网络断开时：

```text
本地 llama.cpp 模型 -> 每个用户请求至多一次低风险手机工具调用
```

离线模式不会启动复杂的主子 agent 循环。服务端只暴露 `observe`、`tap`、`back`、`home`、`wait`、`interact` 和 `take_over`，并拒绝同一请求中的第二次或并行手机动作。多步骤、高风险或不确定任务仍会停止并要求用户确认或交还给更强模型。

手机子 agent 环境变量：

- `PHONE_SUBAGENT_MODEL`: 子 agent 模型名称；为空时复用主云端模型。
- `PHONE_SUBAGENT_BASE_URL`: 可选 OpenAI-compatible 子模型地址。
- `PHONE_SUBAGENT_API_KEY`: 子模型 API key；为空时回退 `OPENAI_API_KEY`。
- `PHONE_SUBAGENT_MAX_TOKENS`: 子模型最大输出 token，默认 `2048`。
- `PHONE_SUBAGENT_MAX_TOOL_CALLS`: 普通 TODO 最大手机工具调用数，默认 `1`。
- `PHONE_SUBAGENT_SHORT_CHAIN_MAX_TOOL_CALLS`: 确定性短链最大手机工具调用数，默认 `4`。

`PHONE_SUBAGENT_BASE_URL` 只能指向可信服务。子 agent 为执行页面操作会接收最新截图和 UI 树；服务端会在调用子模型前拒绝包含密码、token、cookie、session 或授权头等敏感值的 TODO，并返回 `needs_user_action` 交由用户接管。

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

本地模型模式会额外注入一份更保守的系统提示词，只允许简单任务、每个用户请求零次或一次低风险手机工具调用；多步骤、高风险或不确定任务会停止并要求用户确认或交还给更强模型。

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
