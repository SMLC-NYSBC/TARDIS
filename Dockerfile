FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
LABEL name="TARDIS-em"
LABEL authors="robertkiewisz"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget ca-certificates &&\
    rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.11
ENV NAME=tardis

RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda create -y --name $NAME python=$PYTHON_VERSION numpy pandas scikit-learn &&\
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/conda/envs/$NAME/bin:$PATH
RUN conda install --name $NAME pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# setup tardis-em install directory
WORKDIR /opt/tardis-em
COPY . .

# now install
RUN pip install -v .