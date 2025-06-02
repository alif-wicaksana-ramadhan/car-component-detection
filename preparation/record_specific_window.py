import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import sys


def record_screen(window_name: str, output_video: str):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 30.0
    record_seconds = 20

    w = gw.getWindowsWithTitle(window_name)[0]
    w.activate()

    out = cv2.VideoWriter(output_video, fourcc, fps, tuple(w.size))

    for i in range(int(record_seconds * fps)):
        img = pyautogui.screenshot(region=(w.left, w.top, w.width, w.height))
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[200:900, 100:950]
        out.write(frame)
        cv2.imshow("screenshot", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    out.release()


if "__main__" == __name__:
    window_name = sys.argv[1]
    output_video = sys.argv[2]
    record_screen(window_name, output_video)
