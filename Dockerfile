FROM nvidia/cuda:8.0-cudnn6-devel

# nvidia-docker hooks
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# set up environment
ENV DEBIAN_FRONTEND noninteractive

# update repos/ppas...
RUN apt-get update 
RUN apt-get install -y python-software-properties software-properties-common curl
RUN apt-add-repository -y ppa:x2go/stable
RUN apt-get update 

# install core packages
RUN apt-get install -y python-pip git
RUN apt-get install -y python-matplotlib python-scipy python-numpy
RUN apt-get install -y python-opencv

# install python packages
RUN pip install --upgrade pip
RUN pip install --upgrade ipython[all]
RUN pip install pyyaml
# RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
RUN export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
RUN export CUDA_HOME=/usr/local/cuda
RUN pip install tensorflow-gpu
# set up gnuradio and related toolsm
RUN apt-get install -y sudo

# check out sources for reference
RUN mkdir /root/src/

# Gym deps
RUN apt-get install -y python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig pypy-dev automake autoconf libtool

# set up OpenAI Gym
RUN cd /root/src/ && git clone https://github.com/openai/gym.git && cd gym && pip install -e .
RUN pip install gym[atari] pachi_py

# pytorch
RUN pip install -U numpy
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
RUN pip install torchvision

# ROS
# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO kinetic
RUN apt-get update && apt-get install -y \
    ros-kinetic-ros-core=1.3.1-0* \
    && rm -rf /var/lib/apt/lists/*

# Set up environment
RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

# User
RUN useradd -ms /bin/bash user
RUN usermod --password "1234" user
RUN chown user /home/user

# AirSim
RUN apt-get update
RUN apt-get install wget
RUN apt-get install unzip
COPY /AirSim/ /home/user/AirSim/
WORKDIR /home/user/AirSim/
RUN ./setup.sh
RUN ./build.sh

# workspace
RUN mkdir /home/user/workspace
WORKDIR /home/user/workspace

USER user





