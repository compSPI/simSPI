# syntax=docker/dockerfile:1
FROM ubuntu:18.04

WORKDIR /work

RUN apt-get update && \
    apt-get install -y git wget unzip build-essential

RUN wget 'http://www.fftw.org/fftw-3.3.10.tar.gz' && \
    tar -xzvf fftw-3.3.10.tar.gz  && \
    cd fftw-3.3.10 && ./configure && make && make install && \
    cd /work && rm -rf fftw-3.3.10*

RUN wget 'https://iweb.dl.sourceforge.net/project/tem-simulator/TEM-simulator_1.3.zip' && \
    unzip TEM-simulator_1.3.zip && \
    cd TEM-simulator/src && make && \
    cd /work && rm -f TEM-simulator_1.3.zip

ENV PATH=/work/TEM-simulator/src:$PATH
