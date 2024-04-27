FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /workspace/retrosynthesis-prms

COPY ./ .

RUN apt-get update && \
    apt-get install -y --allow-change-held-packages sudo vim curl nano ncdu screen \
    build-essential libsystemd0 libsystemd-dev libudev0 libudev-dev cmake libncurses5-dev libncursesw5-dev git libdrm-dev \
    python3 python3-pip nvtop ncdu 

RUN install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

RUN pip install -r requirements.txt

# download data
RUN sh scripts/download.sh

WORKDIR /workspace/retrosynthesis-prms