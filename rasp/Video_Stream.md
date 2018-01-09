# Video Streaming for Raspberry Pi Zero W

On the host machine:
```bash
# Listen at port and pipe stream through mplayer
nc -l <port> | mplayer -fps 20 -demuxer h264es -

# Or, listen at port and pipe stream to python script
nc -l <port> | python ~/flot/rasp/video_stream_display.py
```


On the Raspberry Pi Zero W
```bash
raspivid -t 0 -w 640 -h 480 -hf -fps 40 -o tcp://<host_ip>:<port>
```

Be sure to start listening at the host before forwarding the stream from the Rapsberry Pi, and that they are communicating at the same port.
