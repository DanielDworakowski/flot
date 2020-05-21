# flōt

https://danieldworakowski.github.io/meetFibi/

Compared to other autonomous platforms, a blimp provides excellent maneuverability and safety for indoor environments due to its low inertia and inherent stability. Our main goal for the project is to build a light-weight blimp and implement a local collision avoidance algorithm, and later combine it with a global planner. One of the main challenges with this project is the significant weight limitations placed on the payload of the blimp. In the design, the sensor payload is limited to a camera along with sonar sensors. Several possible solutions were explored, including traditional methods involving mapping and object detection, as well as solutions like end-to-end learning for collision avoidance. 

## Project Description 

Given the run-time constraints and the desire for functionality in new environments with minimal labeling approaches involving algorithms like SLAM, or scene segmentation were avoided. On the other hand, the end-to-end learning approach allows for automatically labeled training data based on inputs at the time of collection, along with more expressive and generalized features. An interesting aspect of the project is the use of simulated data to first pre-train the network to reduce the need of real data. Given the nature of the problem, the DAGGER algorithm works well to alleviate distribution mismatches and helps to improve the learned policy. In the future we plan to implement deep reinforcement learning methods to tackle the problem with methods similar to CAD2RL and Cognitive Mapping and Planning for Visual Navigation.

## The neural network navigating within an environment
<div style="text-align:center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=KOFQP3Pj4PY
" target="_blank"><img src="http://img.youtube.com/vi/KOFQP3Pj4PY/maxresdefault.jpg" 
alt="" width="640" height="360" border="0" style="text-align:center" /></a>
</div>

Data collected based on https://arxiv.org/abs/1704.05588

## An older version of the platform 
<div style="text-align:center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=krgU84V8UmE
" target="_blank"><img src="http://img.youtube.com/vi/krgU84V8UmE/maxresdefault.jpg" 
alt="" width="640" height="360" border="0" style="text-align:center" /></a>
</div>

# Installed Software

- Tensorflow
- PyTorch
- OpenAI Gym
- ROS
- AirSim

Blocks enviroment is included as a packaged version. If full install of AirSim and Unreal Engine/Editor is required, visit:
https://hub.docker.com/r/raejeong/robotics_ws/

# Install

Install docker

Install nvidia-docker

Clone this repo

```cd flot_ws```

Copy the LinuxNoEditor packaged enviroment in the /home/user/workspace/SimulationEnvironments

Edit /opt/ros/kinetic/etc/ros/python_logging.conf on Rpi to remove logging

# Run
```docker build . -t flot_ws ```

```docker stop flot; docker rm flot; nvidia-docker run -it --ipc=host --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $(pwd)/workspace:/home/user/workspace --privileged --net=host --name flot flot_ws```

- to run additional terminal 

```docker exec -it flot bash```

- to kill container

```docker stop flot; docker rm flot```

- useful docker commands

docker system prune --all -f

docker save -o <save image to path> <image name>

docker load -i <path to image tar file>

# Data collection on the rasp pi
1. ```rosmaster --core``` on pi
2. ```roslaunch blimp_control datacollect.launch``` on pi (check if camera_stream.sh script's IP is your ip)
3. Copy data to host from ~/.ros/<date-and-time>
4. ```ffmpeg -i video.h264 image_%06d.png``` to extract training data
5. ```python flot/workspace/tools/blimp_data_postprocessing.py --file <new-file>```
6. ```python flot/workspace/tools/curator/curator.py --path <new-file>```

The data from step 2 will be saved in ~/.ros in a folde

The data here needs to be postprocessed:
Run blimp_data_postprocessing.py --files=<name of folder in .ros e.g. 20180201_0203020>
The out.csv will be outputted to the folder in .ros

Further postprocessing may be required
