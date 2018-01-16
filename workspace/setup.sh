#!/usr/bin/env bash
#
# Setup paths.
export LOCAL_HOME=$PWD
echo export PYTHONPATH=$PWD/../external/AirSim/PythonClient:"$(find $LOCAL_HOME/ -maxdepth 1 -type d -not -path '*__pycache__*' | sed '/\/\./d' | tr '\n' ':' | sed 's/:$//')" >> $HOME/.bashrc
echo export PYTHONPATH=\$PYTHONPATH:../external/pytorch-cnn-visualizations/src >> $HOME/.bashrc
source $HOME/.bashrc
#
# Initialize the airsim sub-module.
git submodule update --init --recursive
#
# Initialize airsim.
cd ../external/AirSim
./setup.sh
./build.sh
cd $LOCAL_HOME
#
# Install the rate limiter python package.
sudo pip install ratelimiter
sudo pip install visdom
sudo pip install tensorboardX
sudo pip install tensorflow-tensorboard
sudo pip install interval_tree
sudo pip install tqdm
sudo pip install pandas
sudo pip install pathlib
sudo pip install matplotlib
sudo pip install scikit-image



# pip install ratelimiter
# pip install visdom
# pip install tensorboardX
# pip install tensorflow-tensorboard
# pip install interval_tree
# pip install tqdm
