#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

import argparse
import os

import cv2


def main(vid_path: str, output_path: str, happy: bool = False):
    # Open the video with OpenCV:
    vid = cv2.VideoCapture(vid_path)
    # Get the number of frames:
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the frame rate:
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    # Get the width and height of the video:
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Load all the frames in a list:
    frames = []
    for i in range(n_frames):
        ret, frame = vid.read()
        frames.append(frame)

    # Create a window to display the frames:
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", width // 2, height // 2)
    # Use left to go back and right to go forward:
    i = 0
    start, end = None, None
    while True:
        cv2.imshow("Frame", frames[i])
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == ord("l"):
            i = min(i + 1, n_frames - 1)
        elif key == ord("h"):
            i = max(i - 1, 0)
        elif key == ord("s"):
            print(f"Recoding from frame {i}")
            start = i
        elif key == ord("e"):
            print(f"Recording until frame {i}")
            end = i
    cv2.destroyAllWindows()
    if start is None or end is None:
        print("No start or end frame selected, exiting...")
        return
    print(f"[*] Exporting frames {start} to {end}...")
    # Save the frames in the output directory:
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"[*] Output directory {output_path} created.")
    else:
        print(f"[*] Output directory {output_path} already exists.")
        print(f"[*] Overwriting...")
    for i, frame in enumerate(frames[start:end]):
        i = i + start
        label = i
        # i starts at 0 but we want to rescale to [0, 1] using min-max:
        label = (label - start) / ((end - 1) - start)
        if os.path.isfile(os.path.join(output_path, f"0.jpg")):
            label = 00
        label = label if happy else -label
        cv2.imwrite(os.path.join(output_path, f"{label:.02f}.jpg"), frame)


parser = argparse.ArgumentParser()
parser.add_argument("vid_path", help="Path to the video file")
parser.add_argument("-o", "--output", help="Path to the output directory")
parser.add_argument("--happy", action="store_true", help="Label happy faces")
args = parser.parse_args()
main(args.vid_path, args.output, args.happy)
