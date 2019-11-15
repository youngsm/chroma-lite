# Chroma: Ultra-fast Photon Monte Carlo

Chroma is a high performance optical photon simulation for particle physics detectors originally written by A. LaTorre and S. Seibert. It tracks individual photons passing through a triangle-mesh detector geometry, simulating standard physics processes like diffuse and specular reflections, refraction, Rayleigh scattering and absorption.

With the assistance of a CUDA-enabled GPU, Chroma can propagate 2.5 million photons per second in a detector with 29,000 photomultiplier tubes. This is 200x faster than the same simulation with GEANT4.

Information about the historical development of Chroma can be found at https://chroma.bitbucket.io/index.html and https://chroma.bitbucket.io/_downloads/chroma.pdf.

This repository contains a modified version of Chroma that is updated for Python3, Geant4.10.05.p01, and Root 6.18.04, among other things.

## Installation

The `installation` directory contains a `Dockerfile` to build an ubuntu-derived image containing Chroma. There is also an `ubuntu_install.sh` script that will perform an analogous installation on an ubuntu host, and may be a useful reference for other systems. Note that properly linking to boost_python and boost_numpy is nontrivial.

## Usage

Coming soon!
