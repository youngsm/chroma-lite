if [ -e /opt/anaconda3/bin/conda ]; then
    eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
    export LD_LIBRARY_PATH="/opt/anaconda3/lib:$LD_LIBRARY_PATH"
fi
if [ -e /opt/boost ]; then
    export LD_LIBRARY_PATH="/opt/boost/lib:$LD_LIBRARY_PATH"
    export PATH="/opt/boost/bin:$PATH"
    export BOOST_ROOT="/opt/boost"
fi
if [ -e /opt/root/bin/thisroot.sh ]; then
    source /opt/root/bin/thisroot.sh
fi
if [ -e /opt/geant4/bin/geant4.sh ]; then
    source /opt/geant4/bin/geant4.sh
    export PYTHONPATH="/opt/geant4/lib:$PYTHONPATH"
fi
