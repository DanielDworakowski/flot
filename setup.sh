#!/usr/bin/env bash
#
# Setup paths.
echo export PYTHONPATH=$PWD/util:$PWD/agents:$PWD/configs:$PWD/environment:$PWD/external/AirSim/PythonClient:\$PYTHONPATH >> $HOME/.bashrc
export LOCAL_HOME=$PWD
source $HOME/.bashrc
#
# Initialize the airsim sub-module.
git submodule update --init --recursive
#
# Initialize airsim.
cd external/AirSim
./setup.sh
./build.sh
cd $LOCAL_HOME
#
# Install the rate limiter python package.
pip install ratelimiter
