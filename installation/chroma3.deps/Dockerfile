FROM ubuntu:20.04 as build-anaconda
MAINTAINER Benjamin Land <benland100@gmail.com>

USER root
RUN apt-get update 
RUN apt-get install -y curl git

WORKDIR /opt/
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
RUN chmod +x Anaconda3-2019.10-Linux-x86_64.sh
RUN ./Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3
RUN rm Anaconda3-2019.10-Linux-x86_64.sh
COPY chroma_env.sh /etc/profile.d/
SHELL ["/bin/bash", "-l", "-c"] 
RUN conda install -y -c anaconda boost cmake


FROM ubuntu:20.04 AS build-root
MAINTAINER Benjamin Land <benland100@gmail.com>

USER root
RUN apt-get update 
RUN apt-get install -y curl git
COPY chroma_env.sh /etc/profile.d/
SHELL ["/bin/bash", "-l", "-c"] 
COPY --from=build-anaconda /opt/anaconda3 /opt/anaconda3
RUN apt-get install -y dpkg-dev gcc g++ binutils libx11-dev libxpm-dev libxft-dev libxext-dev libgsl-dev libfftw3-dev libpcre3-dev

WORKDIR /opt/
RUN curl -O https://root.cern/download/root_v6.18.04.source.tar.gz
RUN tar xf root_v6.18.04.source.tar.gz
RUN rm root_v6.18.04.source.tar.gz
WORKDIR /opt/root-6.18.04/build-root
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/root -Dminuit2=ON -Droofit=ON /opt/root-6.18.04
RUN make -j12
RUN make install
RUN rm -rf /opt/root-6.18.04

FROM ubuntu:20.04 as build-geant4
MAINTAINER Benjamin Land <benland100@gmail.com>

USER root
RUN apt-get update 
RUN apt-get install -y curl git
COPY chroma_env.sh /etc/profile.d/
SHELL ["/bin/bash", "-l", "-c"] 
COPY --from=build-anaconda /opt/anaconda3 /opt/anaconda3
RUN apt-get install -y dpkg-dev gcc g++ binutils libx11-dev libxpm-dev libxft-dev libxext-dev libgsl-dev libfftw3-dev libpcre3-dev
COPY --from=build-root /opt/root /opt/root
RUN apt-get install -y libxerces-c-dev libxmu-dev libxi-dev freeglut3-dev

WORKDIR /opt/
RUN curl -O https://geant4-data.web.cern.ch/geant4-data/releases/geant4.10.05.p01.tar.gz
RUN tar xf geant4.10.05.p01.tar.gz
RUN rm geant4.10.05.p01.tar.gz
WORKDIR /opt/geant4.10.05.p01/build-g4
RUN cmake -DCMAKE_BUILD_TYPE=Release -DGEANT4_INSTALL_DATA=OFF -DCMAKE_INSTALL_PREFIX=/opt/geant4 -DGEANT4_USE_GDML=ON -DGEANT4_USE_OPENGL_X11=ON -DGEANT4_USE_XM=OFF /opt/geant4.10.05.p01
RUN make -j12
RUN make install
WORKDIR /opt/geant4.10.05.p01
COPY g4py.4.10.05.p01.patch ./
RUN git apply g4py.4.10.05.p01.patch
WORKDIR /opt/geant4.10.05.p01/build-g4py
RUN BOOST_ROOT=/opt/anaconda3 cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/geant4 /opt/geant4.10.05.p01/environments/g4py
RUN make -j12 
RUN sed -i 's/install: preinstall/install:/' Makefile
RUN find . -name '*.cmake' -exec sed -i -n -E '/\.pyc|\.pyo/!p' {} \;
RUN sed -i -E 's/(.*G4LossTableManager.Instance.*)/#\1/' source/python3/__init__.py
RUN make install
RUN rm -rf /opt/geant4.10.05.p01


FROM ubuntu:20.04
MAINTAINER Benjamin Land <benland100@gmail.com>

USER root
RUN apt-get update 
RUN apt-get install -y curl git
COPY chroma_env.sh /etc/profile.d/
SHELL ["/bin/bash", "-l", "-c"] 
COPY --from=build-anaconda /opt/anaconda3 /opt/anaconda3
RUN apt-get install -y dpkg-dev gcc g++ binutils libx11-dev libxpm-dev libxft-dev libxext-dev libgsl-dev libfftw3-dev libpcre3-dev
COPY --from=build-root /opt/root /opt/root
RUN apt-get install -y libxerces-c-dev libxmu-dev libxi-dev freeglut3-dev
COPY --from=build-geant4 /opt/geant4 /opt/geant4

CMD ["/bin/bash","-l"]
