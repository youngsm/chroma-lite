FROM benland100/chroma3:nvidia.base
MAINTAINER Benjamin Land <benland100@gmail.com>

WORKDIR /opt/
ADD https://api.github.com/repos/BenLand100/chroma/git/refs/heads/master /opt/chroma_version.json
RUN git clone https://github.com/BenLand100/chroma
WORKDIR /opt/chroma/
RUN sed -i 's/VIRTUAL_ENV/CONDA_PREFIX/g' setup.py #use anaconda env instead
RUN python setup.py develop

#Only needed for runtime, confuses the build process
RUN echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/anaconda3/lib"' >> /etc/profile.d/chroma_env.sh

#For jupyter
EXPOSE 8888

CMD ["/bin/bash","-l"]
