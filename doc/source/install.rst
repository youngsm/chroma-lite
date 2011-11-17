Installation
============

Chroma development tends to live on the bleeding-edge.  Installation
of Chroma requires a more significant hardware and software investment
than other packages, but we think the rewards are worth it!

Hardware Prerequisites
----------------------

At a minimum, Chroma requires:

* An x86 or x86-64 CPU.
* A NVIDIA GPU that supports CUDA compute capability 1.1 or later.

However, Chroma can be quite demanding with large detector geometries.
Both the CPU and GPU will need sufficient memory hold the detector
geometry and related data structures.  For example, a detector
represented with 60.1 million triangles requires 2.2 GB of CUDA device
memory, and more than 6 GB of host memory during detector
construction.  Chroma also requires the use of multiple CPU cores to
generate photon vertices with GEANT4 at a rate sufficient to keep up
with GPU propagation.

We highly recommend that you run Chroma with:

* An x86-64 CPU with at least four cores.
* 8 GB or more of system RAM
* An NVIDIA GPU that supports CUDA compute capability 2.0 or later,
  and has at least 1.25 GB of device memory.  For detectors with more
  than 20,000 photomultiplier tubes, you will need a GeForce GTX 580
  with 3 GB of device memory, or a Tesla C2050 or higher.

.. note:: The Chroma interactive renderer includes optional support for
  the `Space Navigator 3D mouse <http://www.3dconnexion.com/products/spacenavigator.html>`_, which makes it 10x more fun to fly
  through the detector geometry!

Software Prerequisites
----------------------

Chroma depends on several software packages:

* Python 2.6 or later
* The CUDA 4.0 Toolkit and NVIDIA driver. (You may use drivers newer than the developer driver listed on the CUDA 4.0 website.)
* Boost::Python
* Numpy 1.6 or later
* Pygame
* Matplotlib
* uncertainties
* PyCUDA 2011.2 or later
* PyUblas
* ZeroMQ
* GEANT4.9.4 or later
* g4py
* ROOT 5.30 or later

Optional Space Navigator support requires:

* spacenavd (daemon running on the computer with the 3D mouse)
* libspnav (client library on the system running Chroma)
* spnav python module

Space Navigator control works over ssh when X Forwarding is enabled.

For development, we also recommend:

* nose (to run unit tests)
* coverage (to measure the source coverage of the tests)
* pylint (to check for common problems in Python code)
* sphinx 1.1 dev or later (to generate the documentation with mathjax support)

We will explain how to install all of these packages in the following section.

Step-by-Step Installation: Ubuntu 11.04
---------------------------------------

Although Chroma can run on any CUDA-supported Linux distribution or
Mac OS X, we strongly recommend the use of Ubuntu 11.04 for Chroma
testing and development.  For these instructions, we will assume you
are starting with a fresh Ubuntu 11.04 installation.

Steps 1 and 2 will require root privileges, but the remaining steps
will be done in your home directory without administrator
intervention.

.. warning:: There is very little support for CUDA inside virtual machines, so you cannot use VirtualBox/VMWare/Parallels to setup your Chroma environment.  Amazon EC2 is able to virtualize Tesla devices with Xen, but setting that up for yourself is beyond the scope of this document.

Step 1: ``apt-get`` packages from Ubuntu package manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many packages are required to setup your build environment to compile
GEANT4 and ROOT.  Fortunately, they can be installed with one very
long ``apt-get`` line.  Although this line may wrap in your browser,
it should be executed as one line::

  sudo apt-get install build-essential xorg-dev python-dev gcc-4.4 gfortran-4.4 \
       g++-4.4 python-virtualenv python-numpy python-pygame libglu1-mesa-dev \
       glutg3-dev cmake uuid-dev liblapack-dev mercurial git subversion \
       libboost-all-dev

To be able to generate the documentation, we also need these tools::

  sudo apt-get install texlive dvipng

We have installed GCC 4.4 because CUDA 4.0 does not support GCC 4.5,
the default compiler in Ubuntu.  We will modify the environment in
Step 3 to make gcc 4.4 the default compiler while using Chroma.

Step 2: CUDA Toolkit and Driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA requires the use of the official NVIDIA graphics driver, rather
than the open source Nouveau driver that is included with Ubuntu.  The
NVIDIA driver can be installed by

.. warning:: FIGURE THIS OUT WITH FRESH UBUNTU INSTALL

After the driver is installed, you need to download the CUDA 4.0
toolkit for Ubuntu Linux 10.10 (probably 64-bit) on `this page
<http://developer.nvidia.com/cuda-toolkit-40>`_.  Once this file has
been downloaded, run the following commands in the download
directory::

  chmod +x cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
  sudo ./cudatoolkit_4.0.17_linux_64_ubuntu10.10.run

Accept the default installation location ``/usr/local/cuda``.  We will
add the CUDA ``bin`` and ``lib`` directories to the path in a few
steps.


Step 3: virtualenv
^^^^^^^^^^^^^^^^^^

The excellent `virtualenv <http://www.virtualenv.org/>`_ tool
allows you to create an isolated Python environment, independent from
your system environment. We will keep all of the python modules for
Chroma (with a few exceptions) and libraries compiled from source
inside of a virtualenv in your ``$HOME`` directory::

  virtualenv $HOME/chroma_env
  cd $HOME/chroma_env/bin/
  ln -s /usr/bin/gcc-4.4 gcc
  ln -s /usr/bin/g++-4.4 g++
  ln -s /usr/bin/cpp-4.4 cpp
  ln -s /usr/bin/gcov-4.4 gcov
  ln -s /usr/bin/gfortran-4.4 gfortran

Next, append the following lines to the end of
``$HOME/chroma_env/bin/activate`` to add the CUDA tools to the path::

  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH


Finally, we can enable the virtual environment::

  source $HOME/chroma_env/bin/activate

This will put the appropriate version of python in the path and also
set the ``$VIRTUAL_ENV`` environment variable we will use in the
remainder of the directions.

Step 4: ROOT
^^^^^^^^^^^^

Chroma uses the ROOT I/O system to record event information to disk
for access later.  In addition, we expect many Chroma users will
want to use ROOT to analyze the output of Chroma.

Begin by downloading the ROOT 5.30 tarball from `the ROOT download
page <http://root.cern.ch/drupal/content/production-version-530>`_.
As of this writing, the latest version is 5.30.04.  Then, from the
download directory, execute the following commands::

  tar xvf root_v5.30.04.source.tar.gz
  mkdir $VIRTUAL_ENV/src/
  mv root $VIRTUAL_ENV/src/root-5.30.04
  cd $VIRTUAL_ENV/src/root-5.30.04
  ./configure
  make

We also need to append a ``source`` line to ``$VIRTUAL_ENV/bin/activate``::

  source $VIRTUAL_ENV/src/root-5.30.04/bin/thisroot.sh


Step 5: CLHEP and GEANT4
^^^^^^^^^^^^^^^^^^^^^^^^

Chroma uses GEANT4 to propagate particles other than optical photons
and create the initial photon vertices propagated on the GPU.  These
instructions describe how to compile GEANT4 using the new CMake-based
build system, which requires at least GEANT4.9.4.

GEANT4 depends on CLHEP, so first go to the `CLHEP Download Page
<http://proj-clhep.web.cern.ch/proj-clhep/DISTRIBUTION/>`_ and
download the latest 2.1 series source file.  From your download
directory, run the following commands::

  tar xvf clhep-2.1.1.0.tgz
  mv 2.1.1.0 $VIRTUAL_ENV/src/clhep-2.1.1.0
  cd $VIRTUAL_ENV/src/clhep-2.1.1.0/CLHEP
  ./configure --prefix=$VIRTUAL_ENV/
  make install
  
Now go to the `GEANT4 Download Page <http://geant4.cern.ch/support/download.shtml>`_ and download:

* GNU or Linux tar format source code
* Neutron data files with thermal cross sections
* Data files for low energy electromagnetic processes
* Data files for photon evaporation
* Data files for radioactive decay hadronic processes
* Data files for nuclear shell effects
* Data files for evaluated neutron cross-sections on natural composition of elements

(I wish there was an easier way to get GEANT4 and all of its data files!)

Next go to your download directory and run the following commands::

  tar xvf geant4.9.4.p02.tar.gz
  tar xvf G4EMLOW.6.19.tar.gz
  tar xvf G4PhotonEvaporation.2.1.tar.gz
  tar xvf G4RadioactiveDecay.3.3.tar.gz
  tar xvf G4ABLA.3.0.tar.gz
  tar xvf G4NEUTRONXS.1.0.tar.gz
  mkdir geant4.9.4.p02/data
  mv G4EMLOW6.19 PhotonEvaporation2.1 RadioactiveDecay3.3 G4ABLA3.0 G4NEUTRONXS1.0 geant4.9.4.p02/data/
  mv geant4.9.4.p02 $VIRTUAL_ENV/src/
  cd $VIRTUAL_ENV/src/
  mkdir geant4.9.4.p02-build
  cd geant4.9.4.p02-build
  cmake -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV ../geant4.9.4.p02
  make install

GEANT4 requires several environment variables to locate data files.  Set
these by appending the following lines to ``$VIRTUAL_ENV/bin/activate``::

  export G4LEVELGAMMADATA=$VIRTUAL_ENV/geant4.9.4.p02/data/PhotonEvaporation2.1
  export G4LEDATA=$VIRTUAL_ENV/geant4.9.4.p02/data/G4EMLOW6.19
  export G4NEUTRONHPDATA=$VIRTUAL_ENV/geant4.9.4.p02/data/G4NDL3.14
  export G4RADIOACTIVEDATA=$VIRTUAL_ENV/geant4.9.4.p02/data/RadioactiveDecay3.3


Step 6: g4py
^^^^^^^^^^^^

To access GEANT4 from Python, Chroma uses the g4py wrappers.  We have
had to fix a few bugs and add wrapper a few additional classes for
Chroma, so for now you will need to use our fork of g4py::

  cd $VIRTUAL_ENV/src
  hg clone https://bitbucket.org/seibert/g4py
  cd g4py
  export CLHEP_BASE_DIR=$VIRTUAL_ENV
  # select system name from linux, linux64, macosx as appropriate
  ./configure linux64 --prefix=$VIRTUAL_ENV --with-g4-incdir=$VIRTUAL_ENV/include/geant4 --with-g4-libdir=$VIRTUAL_ENV/lib --libdir=$VIRTUAL_ENV/lib/python2.7/site-packages/
  make install

Step 7: Chroma
^^^^^^^^^^^^^^

Finally, we are getting close to being able to use ``pip`` to do the
rest of the installation.  In order for PyUblas to find boost, we have
to create a file in your ``$HOME`` directory called
``.aksetup-defaults.py`` that contains the following lines::

  BOOST_INC_DIR = ['/usr/include/boost']
  BOOST_LIB_DIR = ['/usr/lib64']
  BOOST_PYTHON_LIBNAME = ['boost_python-mt-py27']

Now we at a stage where the automatic dependency resolution features
of ``pip`` can do their magic.  We need to upgrade the distribute
module and install PyUblas prior to installation, but the rest should
be automatic::

  pip install -U distribute
  pip install pyublas
  pip install -e hg+http://bitbucket.org/chroma/chroma#egg=Chroma

Now you can enable the Chroma environment whenever you want by typing
``source $HOME/chroma_env/bin/activate``, or by placing that line in the
``.bashrc`` login script.

