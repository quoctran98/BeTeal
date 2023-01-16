import argparse
import cv2
import os
import time
import numpy as np
from matplotlib import colors
from pathos.multiprocessing import ProcessingPool as Pool

# not perfect but fast enough :)
# i don't like that teal_frames is a global variable but it's the easiest way to pass it to the Pool
def different_frame(frame_idx, threshold=1):
    frame1 = teal_frames[frame_idx]["frame"]
    try:
        frame2 = teal_frames[frame_idx-1]["frame"]
    except IndexError:       
        return True
    f1_avg_col = np.mean(frame1, axis=(0, 1))
    f2_avg_col = np.mean(frame2, axis=(0, 1))
    # seems to be << 1 for the same frame and > 10 for different frames
    return(np.sqrt(np.sum((f1_avg_col - f2_avg_col)**2)) > threshold)

def score_tealness(frame_dict):
    teal_distance = np.array([np.sqrt(np.sum((pixel - TEAL)**2)) for pixel in frame_dict["frame"]])
    frame_dict["teal_distance"] = np.mean(teal_distance)
    return(frame_dict)

# this is all fake :)
def progress_bar(percent, task_name, bar_length=20):
    arrow = "█" * int(percent/100 * bar_length - 1)
    spaces = " " * (bar_length - len(arrow))
    ending = '\n' if percent == 100 else '\r'
    print(f"\r{task_name} [{arrow}{spaces}] {percent:.2f}%", end=ending)

if __name__ == '__main__':
    # parse filepath arguments
    parser = argparse.ArgumentParser(description="Make your BeTeal!")
    parser.add_argument("-i", "--input_file", type=str,
                        default="./BeReal 2022 Video Recap.MP4",
                        help="filepath to the BeReal Recap video")
    parser.add_argument("-o", "--output_path", type=str,
                        default="./",
                        help="path to save the BeTeal video and image")
    parser.add_argument("-c", "--color", type=str,
                        default="teal",
                        help="color to score the BeReal recap against")
    args = parser.parse_args()

    INPUT_FILE = args.input_file
    OUTPUT_PATH = args.output_path
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # convert color (teal) to BGR
    color = colors.to_rgba(args.color)
    # BGR order for some reason
    TEAL  = np.array([int(color[1]*255), int(color[2]*255), int(color[0]*255)])

    start_time = time.time()

    # save all frames before scoring
    progress_bar(0, f"⚠️ Time to Be{args.color[0].upper() + args.color[1:]}. ⚠️")
    capture = cv2.VideoCapture(INPUT_FILE)
    teal_frames = [] # list of dict of (frame, idx, teal_distance)
    frame_idx = 0
    last_frame = None
    while (True):
        success, frame = capture.read()
        if success:
            teal_frames.append({"frame": frame, "idx": frame_idx})
        else:
            break
        frame_idx += 1
    capture.release()

    # remove duplicate frames
    progress_bar(30, f"⚠️ Time to Be{args.color[0].upper() + args.color[1:]}. ⚠️")
    with Pool() as p:
        is_unique_frame = p.map(different_frame, range(len(teal_frames)))
        teal_frames = [f for f, u in zip(teal_frames, is_unique_frame) if u]

    # score tealness
    progress_bar(70, f"⚠️ Time to Be{args.color[0].upper() + args.color[1:]}. ⚠️")
    with Pool() as p:
        teal_frames = p.map(score_tealness, teal_frames)

    # sort by tealest frames
    teal_frames.sort(key=lambda x: x["teal_distance"], reverse=True)

    # save tealest frame
    cv2.imwrite(f"{OUTPUT_PATH}Be{args.color[0].upper() + args.color[1:]}{'' if args.color[-1] == 'e' else 'e'}st.PNG", teal_frames[-1]["frame"])

    # save video
    progress_bar(80, f"⚠️ Time to Be{args.color[0].upper() + args.color[1:]}. ⚠️")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    resolution = teal_frames[-1]["frame"].shape[:2]
    # fps is lower than original video because no frame duplicates
    out = cv2.VideoWriter(f"{OUTPUT_PATH}Be{args.color[0].upper() + args.color[1:]}.MP4", fourcc, 10.0, (resolution[1], resolution[0])) 
    for i, f in enumerate(teal_frames):
        out.write(f["frame"])
        if i == len(teal_frames) - 1: # pause on the last frame!
            for _ in range(19):
                out.write(f["frame"])
    out.release()
    
    progress_bar(100, f"⚠️ Time to Be{args.color[0].upper() + args.color[1:]}. ⚠️")
    print(f"Be{args.color[0].upper() + args.color[1:]} video created at {OUTPUT_PATH}Be{args.color[0].upper() + args.color[1:]}.MP4 in {time.time() - start_time:.2f} seconds.")