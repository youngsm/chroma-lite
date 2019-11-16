# Chroma: Ultra-fast Photon Monte Carlo

Chroma is a high performance optical photon simulation for particle physics detectors originally written by A. LaTorre and S. Seibert. It tracks individual photons passing through a triangle-mesh detector geometry, simulating standard physics processes like diffuse and specular reflections, refraction, Rayleigh scattering and absorption.

With the assistance of a CUDA-enabled GPU, Chroma can propagate 2.5 million photons per second in a detector with 29,000 photomultiplier tubes. This is 200x faster than the same simulation with GEANT4.

Information about the historical development of Chroma can be found at https://chroma.bitbucket.io/index.html and https://chroma.bitbucket.io/_downloads/chroma.pdf.

This repository contains a modified version of Chroma that is updated for Python3, Geant4.10.05.p01, and Root 6.18.04, among other things.

## Installation

The `installation` directory contains a collection of `Dockerfile`s to build an ubuntu-derived image containing Chroma. This may be a useful reference for other systems. Note that properly linking to boost_python and boost_numpy is nontrivial on systems with both python2 and python3.

Containers are great for packaging dependencies, but CUDA throws an additional constraint that the nvidia-driver version inside the container must match the host nvidia-driver version that created nvidia device nodes. This is because the device nodes created by the host are passed directly to the container. Because of this, an image will need to be built for each possible driver version. 

The images pushed to DockerHub are built from the subdirectories in the `installation` directory. `installation/chroma3.deps` was used to build the image `benland100/chroma3:deps` by running `docker build -t benland100/chroma3:deps` and contains all dependencies for chroma except nvidia-drivers, and can be used to build images for particular versions. `installation/chroma` builds an image using the default version of nvidia-drivers for Ubuntu-20.04 and is pushed to `benland100/chroma3:latest`. The remaining directories create the images `benland100/chroma3:440`, `benland100/chroma3:435`, and `benland100/chroma3:390` for other common versions of nvidia-drivers. If you need another version for your host machine, you will have to create an analogous `Dockerfile`. 

To get a prebuilt image, run `docker pull benland100/chroma3:[tag]` where tag identifies your driver version. 

Geant4 data is not included in these images. You should mount some host directory to /opt/geant4/share/Geant4-10.5.1/data/ when launching a container, and run `geant4-config --install-datasets` to download the data on the first run. 

`docker run -v /path/to/host/data:/opt/geant4/share/Geant4-10.5.1/data/ benland100/chroma3:440 geant4-config --install-datasets`

## Usage

To use CUDA within a container, the host's nvidia device nodes must be passed to the container. To see these devices run `grep /dev/*nvidia*`. Each must be passed to docker with the `--device` flag as shown with `for dev in /dev/*nvidia*; do echo --device $dev:$dev; done`

On my machine, this results in a very concise docker run command:

`docker run -v /home/benland100/research/chroma_test/geant4.10.05.p01/share/Geant4-10.5.1/data/:/opt/geant4/share/Geant4-10.5.1/data/ --device /dev/nvidia-modeset:/dev/nvidia-modeset --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl -it benland100/chroma3:440`

To do something useful, you should mount some host directory containing your analysis code and/or data storage to the container with additional `-v host_path:container_path` flags, and work within those directories. Paths not mounted from the host will not be saved when the container exits.

Consider adding `--net=host` to your run command and running `jupyter` within the container. The default Python3 environment is setup for Chroma.

More coming soon!
