FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

RUN apt-get update && apt-get -y install wget vim-gtk python3-pip python3-pip-whl libgl1-mesa-glx libglfw3-dev libgles2-mesa-dev python3-venv software-properties-common libgeos-dev
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update 

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=/opt/conda/bin:$PATH

COPY ./src hyperdiffusion

RUN pip install -r hyperdiffusion/requirements.txt

# RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html	

