# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90
# From https://github.com/TRAILab/PDV/blob/main/docker/Dockerfile

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.8 (apt)
# pytorch       1.9 (pip)
# ==================================================================
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

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
        libxrender-dev \
        libssl-dev

# ==================================================================
# python
# ------------------------------------------------------------------
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

# ==================================================================
# conda
# ------------------------------------------------------------------

#RUN mkdir -p /opt/conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH
RUN rm -rf ~/miniconda.sh
ENV PATH /opt/conda/envs/ssl/bin:$PATH

RUN /opt/conda/bin/conda init bash \
    && . ~/.bashrc \
    && conda create -n ssl python=3.8 \
    && conda activate ssl 
RUN echo "source activate ssl" > ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "ssl", "/bin/bash", "-c"]


# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# Install cmake v3.21.3
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://cmake.org/files/v3.21/cmake-3.21.3.tar.gz && \
    tar -xzvf cmake-3.21.3.tar.gz && \
    cd cmake-3.21.3 && \
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
#cuda home env needed for minkowski
ENV CUDA_HOME="/usr/local/cuda-11.1" 
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y libgl1

RUN apt-get update -y
RUN apt-get install -y libeigen3-dev
RUN pip install pip==22.1.2
RUN python -m pip --no-cache-dir install -r requirements.txt


COPY third_party third_party
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV PYTHONPATH="/usr/lib/python3.8/site-packages/:${PYTHONPATH}"

WORKDIR /DepthContrast
RUN conda install openblas-devel -c anaconda
RUN python -m pip --no-cache-dir install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt install libopenblas-dev -y
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas" --install-option="--force_cuda"  
# RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
RUN python ./third_party/OpenPCDet/setup.py develop
RUN conda install -c conda-forge/label/gcc7 qhull
RUN conda install -c conda-forge -c davidcaron pclpy
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
RUN pip install nuscenes-devkit
RUN pip install opencv-python-headless

RUN mkdir checkpoints &&  \
    mkdir configs &&  \
    mkdir criterions &&  \
    mkdir data &&  \
    mkdir datasets &&  \
    mkdir models &&  \
    mkdir scripts &&  \
    mkdir tools &&  \
    mkdir utils &&  \
    mkdir output && \
    mkdir lib

# RUN cd && git clone https://github.com/isl-org/Open3D 
# RUN cd /root/Open3D && rm util/install_deps_ubuntu.sh
# RUN cp third_party/patchwork-plusplus/install_deps_ubuntu.sh /root/Open3D/util/install_deps_ubuntu.sh
# WORKDIR /Open3D
# # RUN rm util/install_deps_ubuntu.sh
# # COPY /DepthContrast/third_party/patchwork-plusplus/install_deps_ubuntu.sh util/install_deps_ubuntu.sh
# RUN chmod +x /root/Open3D/util/install_deps_ubuntu.sh
# RUN bash /root/Open3D/util/install_deps_ubuntu.sh

# RUN cd /root/Open3D && mkdir build && cd build && cmake .. && make && make install

# WORKDIR /DepthContrast
# RUN cd third_party/patchwork-plusplus && mkdir build && cd build && cmake .. && make && make install
# RUN cp third_party/patchwork-plusplus/build/python_wrapper/*.so third_party/


