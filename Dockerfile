# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90
# From https://github.com/TRAILab/PDV/blob/main/docker/Dockerfile

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.8 (apt)
# pytorch       1.9 (pip)
# ==================================================================
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

# ==================================================================
# tools
# ------------------------------------------------------------------
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
	    nano \
        libx11-dev \
        fish \
        libsparsehash-dev \
        software-properties-common \
        libsm6 \
        libxext6 \
        libxrender-dev

# ==================================================================
# python
# ------------------------------------------------------------------
WORKDIR /DepthContrast
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-distutils \
        python3-apt \
        python3-pip \
        python3-setuptools
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install --upgrade -r requirements.txt

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# Install cmake v3.13.2
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://cmake.org/files/v3.13/cmake-3.13.2.tar.gz && \
    tar -xzvf cmake-3.13.2.tar.gz && \
    cd cmake-3.13.2 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp

WORKDIR /root

# Install Boost geometry
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.68.0/source/boost_1_68_0.tar.gz
RUN tar xzvf boost_1_68_0.tar.gz
RUN cp -r ./boost_1_68_0/boost /usr/include
RUN rm -rf ./boost_1_68_0
RUN rm -rf ./boost_1_68_0.tar.gz

## Install spconv v1.1
#RUN git clone https://github.com/traveller59/spconv.git
#RUN cd ./spconv && git checkout abf0acf30f5526ea93e687e3f424f62d9cd8313a && git submodule update --init --recursive && SPCONV_FORCE_BUILD_CUDA=1 python setup.py bdist_wheel
#RUN python -m pip install /root/spconv/dist/spconv-*-cp38-cp38-linux_x86_64.whl && \
#    rm -rf /root/spconv
#ENV LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/spconv:${LD_LIBRARY_PATH}"

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# ==================================================================
# DepthContrast Framework
# ------------------------------------------------------------------
WORKDIR /DepthContrast
COPY third_party third_party
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV PYTHONPATH="/usr/lib/python3.8/site-packages/:${PYTHONPATH}"
#WORKDIR /DepthContrast/third_party/pointnet2
#RUN python setup.py install
WORKDIR /DepthContrast/third_party/OpenPCDet
RUN python -m pip --no-cache-dir install --upgrade -r requirements.txt
RUN python -m pip --no-cache-dir install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python setup.py develop

WORKDIR /DepthContrast
RUN python -m pip --no-cache-dir install --upgrade -r requirements.txt
RUN python -m pip --no-cache-dir install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN mkdir checkpoints &&  \
    mkdir configs &&  \
    mkdir criterions &&  \
    mkdir data &&  \
    mkdir datasets &&  \
    mkdir models &&  \
    mkdir scripts &&  \
    mkdir tools &&  \
    mkdir utils &&  \
    mkdir output