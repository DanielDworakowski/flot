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

Copy the LinuxNoEditor packaged enviroment in the /home/user/workspace/SimulationEnvironments

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