#import glob
import cv2
import numpy as np

TEAL = np.array([128, 128, 0])
#INPUT_FILE = glob.glob("*.mp4")[0]
INPUT_FILE = "./BeReals/2022_quoc.mp4"
OUTPUT_PATH = "./"

# not perfect but fast enough :)
def different_frame(frame1, frame2, threshold=1):
    if frame1 is None or frame2 is None:
        return True
    f1_avg_col = np.mean(frame1, axis=(0, 1))
    f2_avg_col = np.mean(frame2, axis=(0, 1))
    # seems to be << 1 for the same frame and > 10 for different frames
    return(np.sqrt(np.sum((f1_avg_col - f2_avg_col)**2)) > threshold)

# go through each frame and score the tealness
capture = cv2.VideoCapture(INPUT_FILE)
teal_frames = [] # tuple of (frame, frame_idx, avg_teal_distance)
frame_idx = 0
last_frame = None
print("Calculating tealness...")
while (True):
    success, frame = capture.read()
    if success:
        # there are frame duplicates because the video speed changes
        if different_frame(frame, last_frame):
            teal_distance = np.array([np.sqrt(np.sum((pixel - TEAL)**2)) for pixel in frame])
            avg_teal_distance = np.median(teal_distance)
            teal_frames.append((frame, frame_idx, avg_teal_distance))
            last_frame = frame
    else:
        break
    frame_idx += 1
capture.release()

# sort by tealest frames
teal_frames.sort(key=lambda x: x[2], reverse=True)

# save tealest frame
cv2.imwrite(f"{OUTPUT_PATH}BeTealest.png", teal_frames[-1][0])

# save video
print("Saving video...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
resolution = teal_frames[-1][0].shape[:2]
# fps is lower than original video because no frame duplicates
out = cv2.VideoWriter(f"{OUTPUT_PATH}BeTeal.mp4", fourcc, 10.0, (resolution[1], resolution[0])) 
for frame, frame_idx, avg_teal_distance in teal_frames:
    out.write(frame)
out.release()