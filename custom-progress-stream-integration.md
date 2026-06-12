# Custom Progress Stream And Scenario Acceptance

## Scope

This document records the backend/frontend contract and scenario acceptance requirements for complex task progress. It intentionally reuses the existing LangGraph Agent Server stream endpoint and the frontend SSE parser.

No new transport is introduced.

## Stream Endpoint

Frontend calls the existing LangGraph run stream endpoint:

```http
POST /threads/{thread_id}/runs/stream
```

The request must include `custom` in `stream_mode`:

```json
{
  "stream_mode": ["messages-tuple", "updates", "tasks", "custom"],
  "on_disconnect": "cancel"
}
```

`on_disconnect` must be `cancel`. Agent Server defaults to `continue`, which
allows the old run to keep executing after the frontend stream disconnects.

The Android frontend already sends this stream mode from:

```text
AIGC_Figma_Frontend\app\src\main\java\com\example\blueheartv\chat\AgentServerClient.kt
```

## Custom Progress Payload

Backend emits two kinds of LangGraph `custom` chunks:

- `task_complexity`: task-level complexity classification, sent once near the beginning of a run.
- `task_progress`: step/tool-level progress updates, sent during tool execution.

## Task Complexity Payload

Backend emits this chunk once per run, near the beginning of task handling:

```json
{
  "type": "task_complexity",
  "complexity": "complex",
  "trackSteps": true,
  "reason": "multi_step_plan_required"
}
```

Required fields:

- `type`: must be `task_complexity`.
- `complexity`: `simple` or `complex`.
- `trackSteps`: `false` for pure text-only answers or tasks expected to need 0/1 tool call, `true` for tasks that need planning into multiple dependent steps.
- `reason`: currently `text_only_answer`, `zero_or_one_tool_call`, or `multi_step_plan_required`.

Classification rule:

- `simple`: pure text answer, or a task expected to complete with 0/1 tool call. Examples include one weather lookup, one system read, or one phone observation.
- `complex`: a task that must be planned into multiple dependent steps or is expected to use more than one tool/action/tool domain.

Examples:

- `解释一下 LangGraph 是什么` -> `simple`, `trackSteps=false`.
- `查询深圳今天的天气` -> `simple`, `trackSteps=false`.
- `读取当前手机已安装应用列表` -> `simple`, `trackSteps=false`.
- `观察当前手机页面` -> `simple`, `trackSteps=false`.
- `查询深圳天气，并读取当前手机已安装应用列表` -> `complex`, `trackSteps=true`.
- `观察当前手机页面，并点击屏幕上的搜索框` -> `complex`, `trackSteps=true`.
- `打开浏览器，搜索蓝心小V` -> `complex`, `trackSteps=true`.

## Task Progress Payload

Backend emits progress chunks shaped as:

```json
{
  "type": "task_progress",
  "label": "tap",
  "status": "running",
  "phase": "phone_tool",
  "message": "Running phone tool: tap",
  "toolName": "tap",
  "progressKey": "phone-todo-2",
  "currentStep": 2,
  "totalSteps": 5,
  "completedSteps": [
    {
      "index": 1,
      "name": "observe",
      "status": "completed"
    }
  ],
  "error": "optional error summary"
}
```

Required fields:

- `type`: must be `task_progress`.
- `label`: stable display label for the current progress item.
- `status`: one of `started`, `running`, `completed`, `failed`.
- `phase`: broad source such as `phone_tool`, `system_tool`, `external_tool`, or `agent`.

Optional fields:

- `message`: human-readable detail.
- `toolName`: stable tool key. Frontend uses it to update the same row across progress chunks.
- `progressKey`: optional stable row identity. Frontend should prefer it when present and fall back to `toolName` for older payloads.
- `currentStep` / `totalSteps`: complex-task step counter when known.
- `completedSteps`: completed step summaries when the backend has a higher-level plan.
- `error`: short failure reason.

## Backend Emission Points

Backend helper:

```text
SuperBlueLM-server\mobile_agent\progress.py
```

Current wrapped tool groups:

- main-agent phone TODO delegation: `mobile_agent/agent/phone_delegation.py`
- phone tools: `mobile_agent/tools/phone.py`
- system tools: `mobile_agent/tools/system.py`
- external tools: `mobile_agent/tools/external.py`

The helper uses LangGraph `get_stream_writer()`. If a tool runs outside a stream context, progress emission is skipped and the tool continues normally.

### Main-Agent TODO Progress

Cloud-mode phone UI operations are delegated through `execute_phone_todo`.
Delegation emits `task_progress` with `phase=agent` before and after each phone
TODO. Atomic `phone_tool` progress is still emitted inside the child execution.

The main agent can append a corrected TODO after observing a failure or an
unexpected page:

- `progressKey` identifies one TODO row, for example `phone-todo-2`.
- `currentStep` is the TODO currently executing.
- `totalSteps` is the number of TODO items currently known by the backend. It may
  grow while the main agent extends or corrects its plan.
- `completedSteps` lists completed TODO summaries.
- A failed TODO remains visible when a corrected TODO is appended.

## Frontend Consumption Points

Frontend model/event files:

```text
AIGC_Figma_Frontend\app\src\main\java\com\example\blueheartv\chat\ChatStreamEvent.kt
AIGC_Figma_Frontend\app\src\main\java\com\example\blueheartv\model\Message.kt
```

Frontend parser:

```text
AIGC_Figma_Frontend\app\src\main\java\com\example\blueheartv\chat\AgentServerClient.kt
```

Frontend state/UI:

```text
AIGC_Figma_Frontend\app\src\main\java\com\example\blueheartv\viewmodel\ChatViewModel.kt
AIGC_Figma_Frontend\app\src\main\java\com\example\blueheartv\ui\components\ChatBubble.kt
```

Frontend behavior:

- `status=running` shows an active tool/progress row.
- `status=completed` marks that row complete.
- `progressKey` should be used as the row identity when present.
- `currentStep/totalSteps` shows a compact counter such as `2/5`.
- `status=failed` keeps the row visible with failure detail.

## Scenario Acceptance Preconditions

The following scenarios assume deployment and connection are already complete.

Preconditions:

- Backend Agent Server is already running.
- Android frontend is already configured to the backend Agent Server address.
- Phone-control client is already connected to the backend `/adb` WebSocket.
- System service is already connected to the backend `/system` WebSocket when the scenario requires system tools.
- Frontend stream requests already include `custom` in `stream_mode`.
- Each run should receive at most one `task_complexity` chunk before or near the first model/tool activity.

The scenarios below do not verify deployment, port availability, WebSocket connection setup, or health-check endpoints. They verify agent behavior after the system is connected.

## Scenario Acceptance Requirements

### Scenario 1: Phone Click

Purpose: verify the agent can observe the current phone screen, decide a visible target, perform a tap action, and stream progress for each completed step.

Suggested prompt:

```text
观察当前手机页面，点击屏幕上可见的搜索框。
```

Expected agent steps:

1. Understand that the task requires phone-screen operation.
2. Call `observe` to get the current screenshot and UI tree.
3. Locate a visible search box or search-like input target from the observation result.
4. Call `tap` with the target coordinate.
5. Observe or summarize the action result.

Required tools:

- `observe`
- `tap`

Expected task complexity state:

- Initial state: `task_complexity` should be `complex` because the task requires at least two dependent actions: observe the screen and tap the target.
- `trackSteps` should be `true`.
- During execution: current step should advance from observation to tap.
- Final state: task should be marked complete after the phone reports the tap result.

Expected progress stream:

- `task_progress` chunk for `observe` with `status=running`.
- `task_progress` chunk for `observe` with `status=completed`.
- `task_progress` chunk for `tap` with `status=running`.
- `task_progress` chunk for `tap` with `status=completed`.
- If step counters are available, the frontend should be able to show `1/2` after `observe` and `2/2` after `tap`.

Expected result:

- The tapped screen target receives the click action on the phone.
- The final assistant response describes the completed action or the observed result.
- The frontend progress area shows both completed steps.

Completion indicators:

- The phone screen focus changes to the search box or equivalent visible target.
- The streamed progress includes completed `observe` and `tap` steps.
- Completed steps / total steps match the expected two-step flow when counters are present.
- The assistant does not claim completion before `tap` completes.

### Scenario 2: System Tool Read

Purpose: verify the agent can use a system-side read tool and stream progress for the tool call without requiring phone-screen interaction.

Suggested prompt:

```text
读取当前手机已安装应用列表，并告诉我是否包含浏览器类应用。
```

Expected agent steps:

1. Understand that the task requires reading installed application metadata.
2. Call `list_apps`.
3. Inspect returned package/app labels for browser-like applications.
4. Summarize whether a browser-like app is present.

Required tools:

- `list_apps`

Optional tools:

- `get_location`, only when the prompt explicitly asks for location.

Expected task complexity state:

- Initial state: `task_complexity` should be `simple` because the task is expected to need one system read tool.
- `trackSteps` should be `false`.
- During execution: `list_apps` progress may still be streamed, but step tracking is not required.
- Final state: task should be marked complete after the system tool result is summarized.

Expected progress stream:

- `task_progress` chunk for `list_apps` with `phase=system_tool` and `status=running`.
- `task_progress` chunk for `list_apps` with `phase=system_tool` and `status=completed`.
- Step counters are optional and not required for this simple one-tool scenario.

Expected result:

- Assistant response summarizes the returned app information instead of inventing the app list.
- Frontend shows a completed `list_apps` progress row.

Completion indicators:

- The agent calls `list_apps`.
- The progress stream contains `phase=system_tool`.
- The initial `task_complexity` chunk uses `complexity=simple` and `trackSteps=false`.
- The assistant response is based on the `list_apps` result.

### Scenario 3: External Tool Read

Purpose: verify the agent can use an external read tool and stream progress while producing a final answer from the tool result.

Suggested prompt:

```text
查询深圳今天的天气，并给出一句出门建议。
```

Expected agent steps:

1. Understand that the task requires weather lookup.
2. Call `weather_query` with city `深圳`.
3. Read the weather result.
4. Produce a short practical recommendation.

Required tools:

- `weather_query`

Optional tools:

- `amap_mcp_tool`, only when the model selects a lower-level whitelisted AMap MCP tool.

Expected task complexity state:

- Initial state: `task_complexity` should be `simple` because the task is expected to need one external lookup tool.
- `trackSteps` should be `false`.
- During execution: weather tool progress may still be streamed, but step tracking is not required.
- Final state: task should be marked complete after the weather result is summarized.

Expected progress stream:

- `task_progress` chunk for `weather_query` or `amap_mcp_tool` with `phase=external_tool` and `status=running`.
- `task_progress` chunk for the same tool with `phase=external_tool` and `status=completed` or `status=failed`.
- Step counters are optional and not required for this simple one-tool scenario.

Expected result:

- Assistant response contains weather information returned from the external tool.
- If `AMAP_MAPS_API_KEY` is missing, the tool output should report the missing environment requirement instead of silently succeeding.
- Frontend shows the external tool progress row and final state.

Completion indicators:

- The agent calls `weather_query` or a whitelisted `amap_mcp_tool`.
- The progress stream contains `phase=external_tool`.
- The initial `task_complexity` chunk uses `complexity=simple` and `trackSteps=false`.
- The progress row reaches completed or failed state.
- When the API key is present, the assistant response includes weather data from the tool result.
- When the API key is missing, the response exposes a clear missing-key requirement.

### Scenario 4: Complex Multi-Step Phone Task

Purpose: verify the agent can execute a multi-step phone task, stream task complexity status, and update completed steps / total steps across several phone actions.

Suggested prompt:

```text
打开浏览器，进入搜索框，输入“蓝心小V”，然后停在搜索结果页。
```

Expected agent steps:

1. Identify that the prompt is a complex phone-control task with multiple dependent actions.
2. Call `observe` to inspect the current phone state.
3. Call `launch` to open the browser if it is not already open.
4. Call `observe` again if needed to confirm the browser page.
5. Call `tap` to focus the search/address box.
6. Call `type` with `蓝心小V`.
7. Call `keyevent` for ENTER or use `tap` on the search action, depending on the UI.
8. Call `wait` if the result page needs loading time.
9. Call `observe` to confirm the final page state.
10. Summarize the final phone state.

Required tools:

- `observe`
- `launch`
- `tap`
- `type`
- `keyevent` or `tap`, depending on how search is submitted.
- `wait`, when the target page needs time to load.

Expected task complexity state:

- Initial state: `task_complexity` should be `complex` because it requires several dependent phone actions.
- `trackSteps` should be `true`.
- During execution: the stream should expose the current action and completed steps.
- Final state: task should be marked complete only after the final page state is confirmed.

Expected progress stream:

- A task-level progress chunk indicates a complex task or equivalent running task state.
- Tool-level progress chunks appear for each phone action.
- Completed steps / total steps should be updated when the total can be planned.
- A reasonable expected step counter is `1/7` through `7/7` when the agent uses: `observe`, `launch`, `tap`, `type`, `keyevent`, `wait`, `observe`.
- If the agent needs an extra `observe` or skips `launch` because the browser is already open, total steps may differ, but the completed count must remain internally consistent.

Expected result:

- The phone opens or stays in a browser.
- The query text `蓝心小V` is entered.
- The phone reaches the search result page or an equivalent submitted-search state.
- Frontend displays progress rows and step counters while actions are running.
- Assistant final response is grounded in the final phone observation.

Completion indicators:

- The stream includes multiple `task_progress` custom chunks in one run.
- The stream includes a complex task running state or tool sequence sufficient to infer complex progress.
- Completed steps never exceed total steps.
- The last required phone action reaches completed state before the final answer.
- The phone is left on the expected search result or target page.

### Scenario 5: Failure

Purpose: verify tool failures are surfaced as explicit task progress and do not appear as a silent stall.

Suggested prompt:

```text
观察当前手机页面，并告诉我现在打开的是哪个应用。
```

Expected agent steps:

1. Understand that the task requires phone observation.
2. Call `observe`.
3. Receive a tool failure from the existing phone tool path.
4. Report that the phone action cannot be completed.

Required tools:

- `observe`

Expected task complexity state:

- Initial state: `task_complexity` should be `simple` because the task is expected to need one phone observation tool.
- `trackSteps` should be `false`.
- During execution: `observe` should enter running state, but step tracking is not required.
- Failure state: `observe` should transition to failed state if the phone tool cannot complete.

Expected progress stream:

- `task_progress` chunk for `observe` with `status=running`.
- `task_progress` chunk for `observe` with `status=failed`.
- The failed chunk includes an `error` value with a short failure reason.

Expected result:

- Frontend keeps the failed progress row visible.
- The assistant response or chat error state clearly indicates that the phone connection is unavailable.

Completion indicators:

- The agent attempts `observe`.
- The initial `task_complexity` chunk uses `complexity=simple` and `trackSteps=false`.
- At least one custom progress payload contains `status=failed`.
- The failed progress payload includes `toolName=observe` or `label=observe`.
- The failed progress row remains visible on the frontend.
- The user can understand why the requested phone observation could not be completed.
