import asyncio
import cv2
import numpy as np
import websockets
import pyautogui
import pygetwindow as gw
import time

SERVER_URL = "ws://localhost:8000/ws/producer"
FPS = 30
QUALITY = 80
WINDOW_NAME = "edge"
CROP_REGION = (200, 900, 100, 950)
SHOW_PREVIEW = True


class WindowCapture:
    def __init__(self):
        self.websocket = None
        self.window = None
        self.running = False

    async def connect(self):
        self.websocket = await websockets.connect(SERVER_URL)
        return True

    def find_window(self):
        windows = gw.getWindowsWithTitle(WINDOW_NAME)
        if not windows:
            return False
        self.window = windows[0]
        self.window.activate()
        return True

    def capture_frame(self):
        img = pyautogui.screenshot(
            region=(
                self.window.left,
                self.window.top,
                self.window.width,
                self.window.height,
            )
        )
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if CROP_REGION:
            top, bottom, left, right = CROP_REGION
            frame = frame[top:bottom, left:right]
        return frame

    def encode_frame(self, frame):
        success, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY]
        )
        return buffer.tobytes() if success else None

    async def send_frame(self, data):
        # if self.websocket and not self.websocket.closed:
        if self.websocket and self.websocket.close_code is None:
            await self.websocket.send(data)
            return True
        return False

    async def start(self):
        if not self.find_window() or not await self.connect():
            return

        self.running = True
        frame_delay = 1.0 / FPS

        while self.running:
            start_time = time.time()

            frame = self.capture_frame()
            if frame is None:
                continue

            if SHOW_PREVIEW:
                cv2.imshow("Capture", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            frame_data = self.encode_frame(frame)
            if frame_data and not await self.send_frame(frame_data):
                print("Failed to send frame")
                if not await self.connect():
                    print("Failed to reconnect")
                    break

            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                print(f"Sleeping for {sleep_time} seconds")
                await asyncio.sleep(sleep_time)

        if self.websocket:
            await self.websocket.close()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


async def main():
    capture = WindowCapture()
    await capture.start()


if __name__ == "__main__":
    asyncio.run(main())
