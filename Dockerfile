FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# change software source
COPY ./docker/sources.list /etc/apt/sources.list
COPY ./docker/.condarc /root/.condarc
COPY ./docker/pip.conf /root/.pip/pip.conf

# apt software
RUN apt-get update && apt-get install -y wget gnuplot git\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# miniconda
WORKDIR /app
ENV PATH="/miniconda3/bin:$PATH"
# installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda-latest.sh \
    && bash ./miniconda-latest.sh -b -p /miniconda3 \
    && ln -s /miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
    && find /miniconda3/ -follow -type f -name '*.a' -delete \
    && find /miniconda3/ -follow -type f -name '*.js.map' -delete \
    && conda clean -afy
# create environment
RUN conda create -n xgnn_env cmake cudnn==8.2.1 python==3.8 \
      pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y \
    && conda clean -afy \
    && echo "conda activate xgnn_env" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "xgnn_env", "/bin/bash", "-c"]

WORKDIR /app/source
COPY . ./xgnn
# install dgl
RUN pip install 'numpy>=1.22' 'scipy>=1.8' 'networkx>=2.7' 'requests>=2.27' \
    && bash ./xgnn/3rdparty/dgl_install.sh
# install fastgraph
RUN bash ./xgnn/utility/fg_install.sh

# install xgnn
RUN pushd ./xgnn \
    && bash ./build.sh \
    && rm -rf build \
    && rm -rf 3rdparty/dgl/build \
    && popd \
    && echo "ulimit -l unlimited" >> ~/.bashrc
