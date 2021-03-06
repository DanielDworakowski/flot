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
RUN apt update
RUN apt-get install -y python3-pip git
RUN apt-get install -y python3-matplotlib python3-scipy python3-numpy

# install python packages
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade ipython[all]
RUN pip3 install pyyaml
RUN pip3 install opencv-python
RUN export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
RUN export CUDA_HOME=/usr/local/cuda
RUN pip3 install tensorflow-gpu
# set up gnuradio and related toolsm
RUN apt-get install -y sudo

# check out sources for reference
RUN mkdir /root/src/

# Gym deps
RUN apt-get install -y python3-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python3-opengl libboost-all-dev libsdl2-dev swig pypy-dev automake autoconf libtool

# set up OpenAI Gym
RUN cd /root/src/ && git clone https://github.com/openai/gym.git && cd gym && pip3 install -e .
RUN pip3 install gym[atari] pachi_py

# pytorch
RUN pip3 install -U numpy
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
RUN pip3 install torchvision

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
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
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
RUN echo 'user:1234' | chpasswd
RUN chown user /home/user
RUN usermod -a -G sudo user

# AirSim
RUN apt-get update
RUN apt-get install wget
RUN apt-get install unzip
WORKDIR /home/user/
RUN apt-get update
RUN apt-get update
RUN git clone https://github.com/Microsoft/AirSim.git
WORKDIR /home/user/AirSim
RUN git pull
RUN apt update
RUN ./setup.sh
RUN ./build.sh
RUN pip3 install msgpack-rpc-python
RUN pip3 install interval_tree
RUN pip3 install ratelimiter
RUN pip3 install visdom
RUN pip3 install tensorboardX
RUN pip3 install tensorflow-tensorboard
RUN pip3 install tqdm
RUN pip3 install pandas
RUN apt-get update
RUN apt-get install -y nano
RUN apt-get install -y ros-kinetic-image-view
RUN apt-get install -y net-tools netcat
RUN pip3 install ai2thor
RUN apt remove -y python3-matplotlib python3-scipy python3-numpy
RUN pip3 install matplotlib scipy numpy
RUN pip3 install scikit-image
RUN mkdir -p /home/user/Documents/AirSim
RUN cp /etc/hosts ~/hosts.new
RUN sed -i -e '1i10.0.1.39       raspberrypi\' ~/hosts.new
RUN cp -f ~/hosts.new /etc/hosts
COPY ./workspace/testEnvs/settings.json /home/user/Documents/AirSim/settings.json
COPY ./workspace/ai2thor/.ai2thor /home/user/.ai2thor

# workspace
RUN mkdir /home/user/workspace
WORKDIR /home/user/workspace
RUN chown -R user /home/user

USER user
RUN echo 'export PYTHONPATH=/home/user/AirSim/PythonClient' >> ~/.bashrc 
RUN echo 'export PYTHONPATH=/home/user/AirSim/PythonClient:${PYTHONPATH}' >> ~/.bashrc 
RUN echo 'export PYTHONPATH=$( find /home/user/workspace/ -type d -printf ":%p" ):${PYTHONPATH}' >> ~/.bashrc
RUN echo 'export ROS_MASTER_URI=http://10.0.1.39:11311' >> ~/.bashrc

