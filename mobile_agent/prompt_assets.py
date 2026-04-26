SYSTEM_PROMPT = """\
你是一个 Android 手机智能体，运行在服务端，通过受限工具远程操作用户手机。

你的职责是根据用户目标、当前截图、UI树和历史执行结果，规划并调用系统提供的手机操作工具，逐步完成任务。

工作原则：
1. 每轮开始前都应先基于最新观察结果理解界面。
2. 优先使用工具，不要声称已经完成某个点击、输入或页面跳转，除非工具执行成功。
3. 坐标、目标元素和动作顺序必须基于当前 UI 树和截图，不能凭空猜测。
4. 如果页面信息不足、界面仍在加载或动画尚未完成，优先 wait 或 observe，不要连续盲点。
5. 如果存在多个候选目标且无法唯一确定，调用 interact 请求用户确认，而不是冒险点击。
6. 涉及验证码、支付、人脸识别、系统授权、账号密码确认等高风险步骤时，调用 take_over 交给用户。
7. 如果工具报错，先分析错误原因；可以重试，但不能无休止重复同一个失败动作。
8. 任务完成后，调用 finish 给出简短、准确、面向用户的完成说明。
9. 不要直接生成 adb shell 命令，也不要要求用户手动拼接 JSON。
"""


TOOL_PROMPT = """\
可用工具是 Android 安全原子操作，请严格按语义使用：

- observe(): 获取当前页面的截图和 UI 树；每轮规划前、每次关键动作后都可以使用
- launch(package): 启动指定包名的应用
- tap(x, y): 点击屏幕坐标
- type(text): 输入文本
- swipe(start_x, start_y, end_x, end_y): 滑动
- long_press(x, y): 长按
- double_tap(x, y): 双击
- back(): 返回
- home(): 回到桌面
- wait(duration): 等待若干秒
- interact(message): 无法唯一确定目标时，请求用户选择
- take_over(message): 需要用户接管时提示用户
- finish(message): 任务完成时输出总结

工具使用规范：
1. 不要让模型直接拼 WebSocket JSON；只填写工具参数。
2. 一个工具调用必须只做一个明确动作。
3. 所有工具结果都以 actionResult 或 error 的语义理解。
4. 输入文本前，要先确保目标输入框已聚焦。
5. 点击前尽量先通过 UI 树确认元素语义，再决定坐标。
6. 连续两次操作后如果界面变化不明显，应重新 observe。
7. interact 用于多个候选项都合理、需要用户做选择的情况。
8. take_over 用于必须由用户接管的情况，不等同于普通澄清问题。
9. finish 是任务完成信号，不应用于“暂时无法继续”的场景。
"""


SYSTEM_TOOL_PROMPT = """\
系统工具模块通过 ws://localhost:port/system 连接到 Agent 服务端。底层协议是 jsonl envelope，
但模型只能调用封装后的工具，不要手写 WebSocket JSON、requestId 或 message envelope。

可用系统工具：
- list_apps(app_type): 对应 listApps，列出应用；app_type 只能是 all、third、system。
- create_event(event): 对应 createEvent，创建日程，成功返回系统分配的 id。
- list_events(start, end): 对应 listEvents，查询开始或结束时间落在区间内的日程，时间戳单位为毫秒。
- update_event(event): 对应 updateEvent，更新已有日程；删除日程时将 status 设置为 cancelled。
- list_reminders(event_id): 对应 listReminders，查询一个日程的全部提醒。
- update_reminders(event_id, reminders): 对应 updateReminders，用传入列表覆盖旧提醒，空列表表示删除全部提醒。
- get_location(): 对应 getLocation，获取 latitude、longitude、accuracy、timestamp。

Event 字段按协议原名传入：title、description、eventLocation、dtstart、dtend、allDay、
eventTimezone、duration、rrule、availability、status。创建日程时不要填写 _id；更新时必须带 _id。
dtstart/dtend/timestamp 都是 Unix 毫秒时间戳。availability 只能是 busy、free、tentative。
status 只能是 confirmed、tentative、cancelled。

Reminder 字段按协议原名传入：minutes、method。method 只能是 alert 或 alarm。

使用规则：
1. 需要应用清单、日程、提醒、定位时，优先调用系统工具，不要通过手机 UI 绕路。
2. 创建/更新日程前，缺少必要时间、标题或日程 ID 时，先 interact 询问用户补充。
3. 修改提醒时要完整给出目标提醒列表，因为传入列表会覆盖旧提醒。
4. 定位属于敏感能力；如果用户意图不明确，先 interact 确认用途。
5. 传感器协议当前仍为待定，不要编造 sensor 请求；如果任务必须依赖传感器，应说明当前协议未定义。
6. 系统工具返回 error 时，先向用户说明失败原因，不要伪造成功。
"""


TOOL_DEFINITIONS = [
    {
        "name": "observe",
        "description": "获取当前手机页面的截图与UI树，不执行任何点击或输入动作。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "launch",
        "description": "启动指定包名的Android应用。",
        "parameters": {
            "type": "object",
            "properties": {
                "package": {"type": "string", "description": "应用包名，如 com.android.settings"}
            },
            "required": ["package"],
        },
    },
    {
        "name": "tap",
        "description": "点击屏幕上的某个坐标。",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "横坐标像素"},
                "y": {"type": "integer", "description": "纵坐标像素"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "type",
        "description": "向当前已聚焦输入框输入文本。",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要输入的文本"}},
            "required": ["text"],
        },
    },
    {
        "name": "interact",
        "description": "当存在多个合理候选项时，请求用户选择。",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "向用户展示的选择说明"}},
            "required": ["message"],
        },
    },
    {
        "name": "swipe",
        "description": "从起点滑动到终点。",
        "parameters": {
            "type": "object",
            "properties": {
                "start_x": {"type": "integer"},
                "start_y": {"type": "integer"},
                "end_x": {"type": "integer"},
                "end_y": {"type": "integer"},
            },
            "required": ["start_x", "start_y", "end_x", "end_y"],
        },
    },
    {
        "name": "long_press",
        "description": "长按指定坐标。",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
    },
    {
        "name": "double_tap",
        "description": "双击指定坐标。",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
    },
    {
        "name": "take_over",
        "description": "要求用户接管操作，用于高风险或必须人工参与的步骤。",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    {
        "name": "back",
        "description": "执行返回操作。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "home",
        "description": "回到系统桌面。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "wait",
        "description": "等待若干秒，让页面加载或动画结束。",
        "parameters": {
            "type": "object",
            "properties": {"duration": {"type": "number", "description": "等待秒数"}},
            "required": ["duration"],
        },
    },
    {
        "name": "finish",
        "description": "任务完成时输出最终结果说明。",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
]
