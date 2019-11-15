if [ -e /opt/anaconda3/bin/conda ]; then
    eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
fi
if [ -e /opt/root/bin/thisroot.sh ]; then
    source /opt/root/bin/thisroot.sh
fi
if [ -e /opt/geant4/bin/geant4.sh ]; then
    source /opt/geant4/bin/geant4.sh
    export PYTHONPATH="/opt/geant4/lib:$PYTHONPATH"
fi
