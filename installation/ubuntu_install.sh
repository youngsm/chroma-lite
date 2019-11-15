#!/bin/bash

#stop on any errors
set -e

#install system dependencies
apt-get update 
apt-get install -y wget git cmake dpkg-dev gcc g++ binutils libx11-dev libxpm-dev libxft-dev libxext-dev libxerces-c-dev libxmu-dev libxi-dev libboost-numpy-dev libboost-python-dev freeglut3-dev nvidia-cuda-toolkit

#install anaconda3
cd $HOME
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
chmod +x Anaconda3-2019.10-Linux-x86_64.sh
./Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
rm Anaconda3-2019.10-Linux-x86_64.sh

#install root
cd $HOME
wget https://root.cern/download/root_v6.18.04.source.tar.gz
tar xf root_v6.18.04.source.tar.gz
rm root_v6.18.04.source.tar.gz
cd root-6.18.04/
mkdir build-root
cd build-root
cmake -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX="/opt/root-6.18.04" -Dminuit2=ON -Droofit=ON "$HOME/root-6.18.04"
make -j12
make install
source /opt/root-6.18.04/bin/thisroot.sh

#cleanup root
cd $HOME
rm -rf root-6.18.04

#install geant4
cd $HOME
wget https://geant4-data.web.cern.ch/geant4-data/releases/geant4.10.05.p01.tar.gz
tar xf geant4.10.05.p01.tar.gz
rm geant4.10.05.p01.tar.gz
cd geant4.10.05.p01
git apply ../g4py.4.10.05.p01.patch
mkdir build-g4
cd build-g4
cmake -DCMAKE_BUILD_TYPE="Release" -DGEANT4_INSTALL_DATA=ON -DCMAKE_INSTALL_PREFIX="/opt/geant4.10.05.p01" -DGEANT4_USE_GDML=ON -DGEANT4_USE_OPENGL_X11=ON -DGEANT4_USE_XM=OFF "$HOME/geant4.10.05.p01"
make -j12
make install
source /opt/geant4.10.05.p01/bin/geant4.sh

#install g4py
mkdir ../build-g4py
cd ../build-g4py
cmake -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX="/opt/geant4.10.05.p01" "$HOME/geant4.10.05.p01/environments/g4py"
make -j12 
#Prevent install from recreating .cmake files
sed -i "s/install: preinstall/install:/" Makefile
#Remove lines trying to find missing bytecode
for f in `find . -name "*.cmake"`; do sed -i -n -E '/\.pyc|\.pyo/!p' $f; done
#Comment out missing G4LossTableManager class instance....
sed -i -E "s/(.*G4LossTableManager.Instance.*)/#\\1/" source/python3/__init__.py
#Install with modified cmake files
make install

#cleanup geant4
cd $HOME
rm -rf geant4.10.05.p01

#install chroma
cd /opt/
git clone https://github.com/BenLand100/chroma
cd chroma
#Ubuntu packages boost for python3 with a suffix
sed -i "s/boost_python/boost_python3/g;s/boost_numpy/boost_numpy3/g" setup.py
python setup.py develop
