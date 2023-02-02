FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel as base

# Install system dependencies
WORKDIR /workdir
ENV LANG C.UTF-8
ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="python -m pip --no-cache-dir install --upgrade --default-timeout 100"
RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL wget tar git
COPY Pipfile Pipfile.lock setup.py /workdir/
RUN conda install --revision 0 &&\
    conda install python==3.8 -y &&\
    conda install -c conda-forge pipenv && \
    # ${PIP_INSTALL} pipenv && \
    pipenv install

# command to run on container start
CMD [ "pipenv", "shell"]
