FROM docker.io/ubuntu:18.04

MAINTAINER Frederic Poitevin "fpoitevi@stanford.edu"

WORKDIR /root

# Install useful packages
RUN apt-get -y update && \
    apt-get -y install git wget vim emacs tmux screen unzip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Download and install FFTW3
RUN wget 'http://www.fftw.org/fftw-3.3.10.tar.gz' && \
    tar -xzvf fftw-3.3.10.tar.gz  && \
    cd fftw-3.3.10 && ./configure && make && make install && \
    cd /root && rm -rf fftw-3.3.10*

# Download and install TEM-simulator
RUN wget 'https://iweb.dl.sourceforge.net/project/tem-simulator/TEM-simulator_1.3.zip' && \
    unzip TEM-simulator_1.3.zip && \
    cd TEM-simulator_1.3/src && make && \
    cd /work && rm -f TEM-simulator_1.3.zip

# Download miniconda installer, perform the installation and remove it
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /root/Miniconda3-latest-Linux-x86_64.sh && \
    /root/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda &&

# Copy the code and generate the environment
COPY environment.yml /root/environment.yml
COPY setup.py /root/setup.py
COPY simSPI /root/simSPI
COPY tests /root/tests
RUN . /opt/anaconda/etc/profile.d/conda.sh && conda update -c conda-forge conda
RUN . /opt/anaconda/etc/profile.d/conda.sh && \
    conda env update --name base --file environment.yml && \
    pip install -e . && \
    conda clean -afy

# Copy and add the entrypoint script
ADD docker/entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]