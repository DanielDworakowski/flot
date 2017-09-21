# flot

Docker Enviroment for Robotics/AI Research

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

```git clone https://github.com/Microsoft/AirSim.git```

Copy the LinuxNoEditor packaged enviroment in the /home/user/workspace/simulation_enviroments

# Run
```sudo docker build . -t flot_ws ```

```sudo nvidia-docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $(pwd)/workspace:/home/user/workspace --privileged --net=host --name flot_ws_c flot_ws```

- to run additional terminal 

```sudo docker exec -it flot_ws_c bash```

- for lazy, to kill all docker

```sudo docker stop $(sudo docker ps -aq); sudo docker rm $(sudo docker ps -aq)```

- useful docker commands

docker system prune --all -f

docker save -o <save image to path> <image name>

docker load -i <path to image tar file>