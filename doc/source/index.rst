.. chroma documentation master file, created by
   sphinx-quickstart on Sat Sep  3 12:36:34 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to chroma's documentation!
==================================

Chroma is a high performance optical photon simulation for particle physics detectors.  It tracks individual photons passing through a triangle-mesh detector geometry, simulating standard physics processes like diffuse and specular reflections, refraction, Rayleigh scattering and absorption.  

With the assistance of a CUDA-enabled GPU, Chroma can propagate 2.5 million photons per second in a detector with 29,000 photomultiplier tubes.  This is 200x faster than the same simulation with GEANT4.

Contents:

.. toctree::
   :maxdepth: 2

   install
   geometry
   render
   simulation
   likelihood

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

