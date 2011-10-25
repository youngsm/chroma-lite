import distribute_setup
distribute_setup.use_setuptools()
from setuptools import setup, find_packages, Extension
import subprocess

def check_output(*popenargs, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd, output=output)
    return output

geant4_cflags = check_output(['geant4-config','--cflags']).split()
geant4_libs = check_output(['geant4-config','--libs']).split()

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
