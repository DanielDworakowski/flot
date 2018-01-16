from __future__ import print_function
from twitchstream.outputvideo import TwitchBufferedOutputStream

from RobotObservation import VideoStreamClient

import argparse
import time
import numpy as np

# Twitch keys for pingu95hk; DO NOT SHARE WITH OTHERS
streamkey = 'live_145306514_ZmQ4sIt4ZAwxNPeLLRpU4YfCdIIHu1'
oauthkey = 'oauth:z1g9zxjfkd0bfb4z8gzdf2y3y7ejmi'

LIVE = False

# Load stream to send the video
with TwitchBufferedOutputStream(
    twitch_stream_key=streamkey,
    width=640,
    height=480,
    fps=40.,
    enable_audio=False,
    verbose=True) as videostream:

    if LIVE:
        client = VideoStreamClient()
        client.start()
    else:
        frame = np.zeros((480, 640, 3))
        frame[:, :, :] = np.array([1, 0, 0])[None, None, :]

    while True:
        # If there are not enough video frames left,
        # add some more.
        if LIVE:
            frame = client.frame

        if (videostream.get_video_frame_buffer_state() < 40) and frame is not None:
            videostream.send_video_frame(frame)
        else:
            time.sleep(0.001)
