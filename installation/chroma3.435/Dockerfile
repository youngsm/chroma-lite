FROM benland100/chroma3:deps
MAINTAINER Benjamin Land <benland100@gmail.com>

USER root
#NVIDIA driver version used here MUST match the host driver version! 
#Container WILL NOT work without this constraint, sorry! No way around it.
#Here I add the nvidia-driver PPA for Ubuntu to get v435 drivers
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN apt-get update && apt-get install -y nvidia-headless-435 nvidia-cuda-toolkit

WORKDIR /opt/
ADD https://api.github.com/repos/BenLand100/chroma/git/refs/heads/master /opt/chroma_version.json
RUN git clone https://github.com/BenLand100/chroma
WORKDIR /opt/chroma/
RUN sed -i 's/VIRTUAL_ENV/CONDA_PREFIX/g' setup.py #use anaconda env instead
RUN python setup.py develop

#For jupyter
EXPOSE 8888

CMD ["/bin/bash","-l"]
