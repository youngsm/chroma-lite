import distribute_setup
distribute_setup.use_setuptools()
from setuptools import setup, find_packages, Extension
import subprocess

geant4_cflags = subprocess.check_output(['geant4-config','--cflags']).split()
geant4_libs = subprocess.check_output(['geant4-config','--libs']).split()

setup(
    name = 'Chroma',
    version = '0.5',
    packages = find_packages(),
    include_package_data=True,

    scripts = ['bin/chroma-sim', 'bin/chroma-cam'],
    ext_modules = [
        Extension('chroma.generator._g4chroma',
                  ['src/G4chroma.cc'],
                  include_dirs=['src'],
                  extra_compile_args=geant4_cflags,
                  extra_link_args=geant4_libs,
                  libraries=['boost_python'],
                  ),
        Extension('chroma.generator.mute',
                  ['src/mute.cc'],
                  extra_compile_args=geant4_cflags,
                  extra_link_args=geant4_libs,
                  libraries=['boost_python']),
        ],
    
    install_requires = ['uncertainties','pyzmq-static','spnav', 'pycuda',
                      'numpy>1.6', 'pygame'],
    test_suite = 'nose.collector',
    
)
