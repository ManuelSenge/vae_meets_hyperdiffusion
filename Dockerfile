FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

# Install base utilities
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y build-essential wget libgl1 libglib2.0-0 freeglut3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate

COPY ./hyperdiffusion_env.yaml /hyperdiffusion_env.yaml

RUN /opt/conda/bin/conda env create -f /hyperdiffusion_env.yaml

ENV PATH=/opt/conda/bin:$PATH

COPY . /hyperdiffusion


# && conda run -n hyper-diffusion  wandb login ${WANDB_API_KEY}
