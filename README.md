# 交互型 WebSocket 服务端

## 功能

- 支持多个客户端连接。
- 服务端终端可实时输入文本并广播给所有客户端。
- 客户端发送的消息会打印在服务端，并转发给所有连接客户端。
- 在服务端输入 `/quit` 可优雅关闭服务。

## 安装依赖

使用你当前的 Python 包管理方式安装依赖（项目已在 `pyproject.toml` 中声明 `websockets`）。

例如：

```bash
pip install -e .
```

## 启动服务

```bash
python main.py --host 127.0.0.1 --port 8765
```

连接地址格式：`ws://127.0.0.1:8765/ws/device/{deviceId}`

## 使用方式

1. 启动服务后，连接 WebSocket 客户端到 `/ws/device/{deviceId}` 路径。
	例如：`ws://localhost:8765/ws/device/device-001`
2. 在服务端终端输入内容并回车，会广播给所有客户端。
3. 输入 `/quit` 停止服务。

{"type":"request","message":"launch","requestId":2,"data":{"package":"com.android.settings"}}
{"type":"request","message":"back","requestId":3,"data":null}
