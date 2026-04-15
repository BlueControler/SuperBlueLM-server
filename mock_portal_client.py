from __future__ import annotations

import asyncio
import json
from typing import Any

from websockets.asyncio.client import connect


class MockPortalClient:
    def __init__(self, device_id: str = "demo-device") -> None:
        self.device_id = device_id
        self.width = 1080
        self.height = 2400
        self.current_package = "com.android.launcher"
        self.current_activity = "Launcher"
        self._running = True
        self._next_request_id = 1

    async def run(self, url: str = "ws://127.0.0.1:8765/ws/devices") -> None:
        async with connect(f"{url}/{self.device_id}") as websocket:
            print(f"[client] connecting to {url}/{self.device_id}")
            await websocket.send(
                json.dumps(
                    {
                        "type": "request",
                        "message": "connect",
                        "requestId": self._next_request_id,
                        "data": {
                            "token": "demo-token",
                            "width": self.width,
                            "height": self.height,
                            "screenshot": None,
                            "ui": self._result("connect")["ui"],
                            "currentPackage": self.current_package,
                            "activity": self.current_activity,
                        },
                    }
                )
                + "\n"
            )
            print("[client] -> connect")
            self._next_request_id += 1

            heartbeat = asyncio.create_task(self._heartbeat_loop(websocket))
            try:
                async for raw in websocket:
                    envelope = json.loads(raw.strip())
                    if envelope["type"] != "request":
                        continue
                    print(
                        f"[client] <- requestId={envelope.get('requestId')} "
                        f"message={envelope['message']} data={envelope.get('data')}"
                    )
                    response = self._handle_server_request(
                        envelope["message"], envelope.get("data")
                    )
                    print(
                        f"[client] -> requestId={envelope.get('requestId')} "
                        f"message=actionResult data={response}"
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response",
                                "message": "actionResult",
                                "data": response,
                                "requestId": envelope.get("requestId"),
                            }
                        )
                        + "\n"
                    )
            finally:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass

    async def _heartbeat_loop(self, websocket) -> None:
        while self._running:
            await asyncio.sleep(1)
            await websocket.send(
                json.dumps({"type": "request", "message": "ping", "data": None}) + "\n"
            )

    def _handle_server_request(self, message: str, data: Any) -> dict[str, Any]:
        if message == "observe":
            return self._result("observe")
        if message == "launch":
            self.current_package = data["package"]
            self.current_activity = "MainActivity"
            return self._result("launch")
        if message in {
            "tap",
            "type",
            "swipe",
            "longPress",
            "doubleTap",
            "back",
            "home",
            "wait",
            "finish",
            "interact",
            "takeOver",
        }:
            return self._result(message)
        return self._result(f"unsupported:{message}")

    def _result(self, action: str) -> dict[str, Any]:
        return {
            "screenshot": None,
            "ui": f"<hierarchy><screen action='{action}' package='{self.current_package}' /></hierarchy>",
            "currentPackage": self.current_package,
            "activity": self.current_activity,
        }


async def main() -> None:
    client = MockPortalClient()
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
