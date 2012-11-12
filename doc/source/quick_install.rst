Quick Installation
==================

Make sure you're computer meets the :ref:`hardware-prerequisites`!

.. _ubuntu11.04_quick:

Ubuntu 11.04
------------

Andy Mastbaum has provided a shell script that downloads and compiles
all of the Chroma prerequisites.  It has been tested to work with
Ubuntu 11.04.  To use this script, first go perform
:ref:`ubuntu_11.04_step1` and :ref:`ubuntu_11.04_step2` in the
Step-by-Step Installation guide.  Then download
:download:`chroma-setup.sh` and run the following::

  chmod +x chroma-setup.sh
  ./chroma-setup -j4 -n ~/chroma_env

This will download and compile (with 4 CPU cores; increase the ``-j``
option if you have more cores) all the source code required to create
a self-contained Chroma environment in ``$HOME/chroma_env``.  To setup
your environment to use Chroma, just run ``source
$HOME/chroma_env/bin/activate``.

.. _rhel6_quick:

Red Hat Enterprise Linux 6
--------------------------

The following procedure has been tested on a basic RHEL6 install.

Disable the nouveau graphics driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Edit /etc/grub.conf and add ``rdblacklist=nouveau`` to the end of the kernel line::

    kernel /vmlinuz-2.6.32-279.11.1.el6.x86_64 ... rdblacklist=nouveau

Create the file /etc/modprobe.d/blacklist-nouveau.conf with the line::

    blacklist nouveau

Install the CUDA Driver and Toolkit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, download the NVIDIA driver appropriate to your machine at the `NVIDIA Unix Drivers Download Page <http://www.nvidia.com/object/unix.html>`_.

Dropt to a full terminal by pressing `CTRL+ALT+F1` and then enter::

    su -
    init 3

Login again at the prompt and enter the following commands::

    su -
    cd /home/user/Downloads
    chmod +x NVIDIA-Linux-x86_64-304.64.run
    ./NVIDIA-Linux-x86_64-304.64.run

During the installation you can just pick all the default options.

Download the CUDA toolkit from the `CUDA Toolkit Archive <https://developer.nvidia.com/cuda-toolkit-41-archive>`_, and run the following commands to install it::

    su -
    cd /home/user/Downloads
    chmod +x cudatoolkit_4.1.28_linux_64_rhel6.x.run
    ./cudatoolkit_4.1.28_linux_64_rhel6.x.run

Download and run `chroma-setup.py <http://chroma.bitbucket.org/_downloads/chroma-setup.py>`_::

    python chroma-setup.py /home/user/chroma_env

where ``/home/user/chroma_env`` is where you would like all of the packages installed. This should preferably be an empty directory.

.. note:: You must be able to run the ``sudo`` command. Near the end of the installation as it is trying to install pygame, the install sort of halts. Just press enter and it will continue.

Now, go get some lunch! The installation takes hours.

If the installation went successfully, chroma and many of the other packages were installed into a `virtual environment <http://www.virtualenv.org/en/latest/>`_ created during the setup. To start using chroma just activate the virtual environment by running::

    source /home/user/chroma_env/bin/activate

Try the following to see if everything worked::

    chroma-cam @chroma.models.lionsolid
