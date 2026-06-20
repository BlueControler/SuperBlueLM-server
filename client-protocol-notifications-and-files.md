# 客户端待实现协议：通知读取与本地文件操作

> 面向 Android 客户端（`AIGC_Figma_Frontend`）开发同学。
>
> 后端已新增四个工具，它们通过现有 `/system` WebSocket 通道下发请求，但**客户端当前尚未实现对应的 message 处理**。本文定义这些 message 的请求/响应结构，供客户端补齐。
>
> 实现位置参考：`app/src/main/java/com/example/blueheartv/system/SystemProtocolHandler.kt`（分发器）+ `SystemApi.kt`（具体能力）。在 `SystemProtocolHandler.handleRequest` 的 `when (message)` 中新增下列 case 即可。

## 通道与信封格式

复用现有 `/system` JSONL 协议，信封格式与现有消息完全一致：

请求（后端 → 客户端）：

```json
{ "type": "request", "message": "<message>", "requestId": 12, "data": { ... } }
```

响应（客户端 → 后端）：

```json
{ "type": "response", "message": "<message>", "requestId": 12, "data": { ... } }
```

失败时把错误放进 `data.error`（与现有实现一致）：

```json
{ "type": "response", "message": "<message>", "requestId": 12, "data": { "error": "权限未授予" } }
```

后端约定：当响应 `data` 是对象且包含 `error` 字段时，视为该工具调用失败，会向用户说明原因，不会伪造成功。

---

## 1. `listNotifications` — 读取系统通知

对应后端工具：`list_notifications`（场景四：睡前生活复盘）。

需要 `NotificationListenerService` 权限（用户在系统设置中授予通知使用权）。未授予时返回 `error`。

### 请求 data

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `since` | long | 否 | Unix 毫秒时间戳下界，只返回该时间之后发布的通知。缺省返回全部当前未清除通知。 |
| `limit` | int | 否 | 返回条数上限（1–100）。 |

`data` 可能为 `null`（两个字段都不传时）。

### 响应 data

返回一个数组，每项为一条通知：

```json
{
  "notifications": [
    {
      "packageName": "com.tencent.mm",
      "appName": "微信",
      "title": "张三",
      "text": "周六下午聚餐，地点你定",
      "postTime": 1748500000000,
      "category": "msg"
    }
  ]
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `packageName` | string | 是 | 发出通知的应用包名。 |
| `appName` | string | 否 | 应用显示名。 |
| `title` | string | 否 | 通知标题。 |
| `text` | string | 否 | 通知正文摘要。 |
| `postTime` | long | 是 | 通知发布时间，Unix 毫秒。 |
| `category` | string | 否 | 通知分类（Android `Notification.category`）。 |

> 权限未授予时返回 `{ "error": "通知使用权未授予" }`，后端会在复盘中注明"未获取通知权限"。

---

## 2. `searchFiles` — 本地文件搜索

对应后端工具：`search_files`（场景一：检索会议参考文档）。

按关键词在设备文档目录中检索文件。建议默认根目录为用户可访问的文档区（如 `Download`、`Documents`、应用共享目录），并遵守 Android 分区存储限制。

### 请求 data

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `keywords` | string[] | 是 | 关键词列表，对文件名或路径做 AND 匹配。 |
| `roots` | string[] | 否 | 限定搜索的根目录绝对路径列表。缺省时使用客户端默认文档根目录。 |
| `limit` | int | 是 | 返回上限（1–100），后端默认 20。 |

### 响应 data

```json
{
  "files": [
    {
      "path": "/storage/emulated/0/Download/产品评审会纪要.pdf",
      "name": "产品评审会纪要.pdf",
      "size": 204800,
      "modifiedTime": 1748400000000,
      "mimeType": "application/pdf"
    }
  ]
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 文件绝对路径。 |
| `name` | string | 是 | 文件名。 |
| `size` | long | 否 | 字节数。 |
| `modifiedTime` | long | 否 | 最后修改时间，Unix 毫秒。 |
| `mimeType` | string | 否 | MIME 类型。 |

无匹配时返回 `{ "files": [] }`（不是错误）。

---

## 3. `archiveFile` — 文件归档（复制/移动）

对应后端工具：`archive_file`（场景一：将参考文档归档到指定目录）。

### 请求 data

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `source` | string | 是 | 源文件绝对路径。 |
| `targetDir` | string | 是 | 目标目录绝对路径，不存在时应创建。 |
| `mode` | string | 是 | `copy`（保留源文件）或 `move`（移动后删除源文件）。 |

### 响应 data

```json
{ "archivedPath": "/storage/emulated/0/Documents/归档/产品评审会纪要.pdf" }
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `archivedPath` | string | 是 | 归档后文件的最终绝对路径。 |

权限/路径错误返回 `{ "error": "..." }`。

---

## 4. `readTextFile` — 读取文本文件

对应后端工具：`read_text_file`（场景一：读取参考文档整理摘要）。

仅支持 UTF-8 文本文件。二进制文件返回错误。

### 请求 data

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 文本文件绝对路径。 |
| `maxBytes` | int | 是 | 最大读取字节数（1–524288），后端默认 65536。超出部分截断。 |

### 响应 data

```json
{
  "path": "/storage/emulated/0/Download/会议纪要.txt",
  "content": "……文件文本内容……",
  "truncated": false,
  "encoding": "utf-8"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 文件路径。 |
| `content` | string | 是 | 文本内容（已按 `maxBytes` 截断）。 |
| `truncated` | bool | 否 | 是否因超过 `maxBytes` 被截断。 |
| `encoding` | string | 否 | 实际编码，建议固定 `utf-8`。 |

---

## 实现核对清单

在 `SystemProtocolHandler.handleRequest` 的 `when (message)` 中补齐：

- [ ] `listNotifications` → 实现 `NotificationListenerService`，读取当前活动通知，按 `since`/`limit` 过滤。
- [ ] `searchFiles` → 在默认文档根目录或 `roots` 下按关键词遍历匹配。
- [ ] `archiveFile` → 按 `mode` 复制或移动文件，必要时创建目标目录。
- [ ] `readTextFile` → 读取 UTF-8 文本并按 `maxBytes` 截断。
- [ ] 所有新增能力在缺权限时返回 `data.error`，不要抛裸异常或伪造成功。
- [ ] 在 `AndroidManifest.xml` 声明所需权限（通知使用权、存储读取/管理）。

实现完成前，后端这四个工具调用会收到 `未知消息类型: <message>` 错误并向用户如实报告，不影响其他场景。
