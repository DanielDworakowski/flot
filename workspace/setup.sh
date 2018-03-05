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
# Install python packages.
pip install ratelimiter
pip install visdom
pip install tensorboardX
pip install tensorflow-tensorboard
pip install interval_tree
pip install tqdm
pip install pandas
pip install pathlib
pip install matplotlib
pip install scikit-image