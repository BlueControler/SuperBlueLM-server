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

    async def run(self, url: str = "ws://127.0.0.1:8765/adb") -> None:
        target_url = self._build_ws_url(url)
        async with connect(target_url) as websocket:
            print(f"[client] connecting to {target_url}")
            await websocket.send(
                json.dumps(
                    {
                        "type": "request",
                        "message": "connect",
                        "requestId": 1,
                        "data": {
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

    def _build_ws_url(self, url: str) -> str:
        if url.endswith("/adb"):
            return url
        if url.endswith("/ws/devices") or url.endswith("/ws/device"):
            return f"{url}/{self.device_id}"
        return url

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
        if message == "listPackages":
            return {
                "packages": [
                    "com.android.settings",
                    "com.android.launcher",
                    "com.example.demo",
                ]
            }
        if message == "keyevent":
            keyevent = data["keyevent"]
            if keyevent == 3:
                self.current_package = "com.android.launcher"
                self.current_activity = "Launcher"
            return self._result(f"keyevent:{keyevent}")
        if message in {
            "tap",
            "type",
            "swipe",
            "longPress",
            "doubleTap",
            "interact",
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
