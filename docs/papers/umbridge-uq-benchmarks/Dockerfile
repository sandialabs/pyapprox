FROM ubuntu:latest

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && \
    apt-get install -y wget git gcc && \
    update-ca-certificates

# COPY Mambaforge.sh .

RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh" -O Mambaforge.sh

RUN bash Mambaforge.sh  -b -p "${HOME}/conda"

RUN git clone https://github.com/sandialabs/pyapprox.git && \
    cd pyapprox && \
    git checkout devel

WORKDIR /pyapprox

RUN source "${HOME}/conda/etc/profile.d/conda.sh"  && \
    conda activate && \
    pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org" && \
    yes | mamba env create -f environment.yml


RUN source "${HOME}/conda/etc/profile.d/conda.sh"  && \
    conda activate pyapprox-base && \
    pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org" && \
    pip install -e . --no-build-isolation

RUN source "${HOME}/conda/etc/profile.d/conda.sh"  && \
    conda activate pyapprox-base && \
    pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org" && \
    pip install umbridge && \
    conda clean -afy && \
    cd ../

WORKDIR /

COPY minimal-server.py /

RUN echo "DONE"

CMD source "${HOME}/conda/etc/profile.d/conda.sh"  && \
    conda activate pyapprox-base && \
    python3 minimal-server.py