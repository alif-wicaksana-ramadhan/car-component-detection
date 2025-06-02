import os
import cv2


def extract_frames(video_path: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(total_frames):
        _, frame = video.read()
        frame = frame[200:900, 100:950]
        frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
        cv2.imwrite(frame_path, frame)

    video.release()
    print(f"Frames extracted and saved to {output_folder}")


if __name__ == "__main__":
    files = os.listdir("./recorded_video")
    for file in files:
        classname = file.split(".")[0]
        video_path = f"./recorded_video/{file}"
        output_folder = f"dataset/{classname}"
        extract_frames(video_path, output_folder)
