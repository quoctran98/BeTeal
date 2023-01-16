# ⚠️ Time to BeTeal. ⚠️

BeTeal reorganizes your BeReal Recap video from less teal frames to more teal frames. 

## Usage

1. Clone this repository and install open-cv

```bash
pip install opencv-python
```

2. Download your BeReal Recap video from the app and save it in the same directory as the script.


3. Run the script. If your video is not named `BeReal 2022 Video Recap.MP4`, add the `-i` flag and specify the filename of your video. The `-o` flag can be used to specify where the output video and image will be saved.

```bash
python3 beteal.py -i "./BeReal 2022 Video Recap.MP4" -o "./"
```

4. The script will output `BeTeal.MP4`, which is the BeTealed Recap, and `BeTealest.PNG`, which is the tealest BeReal in your BeReal Recap video.
