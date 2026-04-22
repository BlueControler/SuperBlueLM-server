from __future__ import annotations

import argparse
import asyncio
import re
import ssl
from contextlib import suppress

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

clients: set[ServerConnection] = set()
PATH_PATTERN = re.compile(r"^(?:/adb|/ws/devices/(?P<device_id>[^/]+)|/ws/device/(?P<legacy_device_id>[^/]+))$")


def _build_ssl_context(certfile: str | None, keyfile: str | None) -> ssl.SSLContext | None:
    if not certfile and not keyfile:
        return None
    if not certfile or not keyfile:
        raise ValueError("Both --certfile and --keyfile are required to enable TLS.")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context


async def broadcast(message: str) -> None:
    if not clients:
        print("[system] 当前没有已连接客户端")
        return

    disconnected: list[ServerConnection] = []
    for ws in clients:
        try:
            print("[broadcast] sending to client:", message)
            await ws.send(message + "\n", text=True)
        except ConnectionClosed:
            disconnected.append(ws)

    for ws in disconnected:
        clients.discard(ws)


async def handle_client(websocket: ServerConnection) -> None:
    request = websocket.request
    if request is None:
        print("[reject] missing websocket request metadata")
        await websocket.close(code=1008, reason="invalid request")
        return

    path = request.path
    match = PATH_PATTERN.match(path)
    if not match:
        print(f"[reject] invalid device path: {path!r}")
        await websocket.close(code=1008, reason="invalid path")
        return

    device_id = match.group("device_id") or match.group("legacy_device_id") or "default"
    clients.add(websocket)
    peer = websocket.remote_address
    print(f"[connect] device {device_id} connected: {peer}")

    try:
        async for message in websocket:
            print(f"[client {device_id}@{peer}] {message}")
    except ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)
        print(f"[disconnect] device {device_id} disconnected: {peer}")


async def console_input_loop(stop_event: asyncio.Event) -> None:
    print("[system] 输入一行 JSON envelope 并回车，可广播给所有客户端")
    print("[system] 输入 /quit 关闭服务")

    while not stop_event.is_set():
        text = await asyncio.to_thread(input, "server> ")
        text = text.strip()

        if not text:
            continue
        if text == "/quit":
            stop_event.set()
            break

        await broadcast(text)


async def run_server(host: str, port: int, certfile: str | None, keyfile: str | None) -> None:
    stop_event = asyncio.Event()
    ssl_context = _build_ssl_context(certfile, keyfile)
    scheme = "wss" if ssl_context else "ws"

    async with serve(handle_client, host, port, ssl=ssl_context):
        print(f"[system] websocket server listening at {scheme}://{host}:{port}/adb")
        input_task = asyncio.create_task(console_input_loop(stop_event))

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            print("\n[system] received Ctrl+C, shutting down...")
        finally:
            input_task.cancel()
            with suppress(asyncio.CancelledError):
                await input_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual websocket JSON console server")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8765, help="Listen port")
    parser.add_argument("--certfile", default=None, help="TLS certificate path for wss")
    parser.add_argument("--keyfile", default=None, help="TLS private key path for wss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_server(args.host, args.port, args.certfile, args.keyfile))
    except KeyboardInterrupt:
        print("\n[system] server exited")


if __name__ == "__main__":
    main()
