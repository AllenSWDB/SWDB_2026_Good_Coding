# hash:sha256:b8cdaf98fb059bd4ee1ce7ea1a6de724e3cfaa2db21357a29e0aab2ddc5929b7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/swdb:Python3.12code-server4.133.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    argparse \
    hdmf==6.1.0 \
    hdmf-zarr==0.13.0 \
    matplotlib==3.11.0 \
    numpy==2.5.1 \
    pandas==3.0.3 \
    pyarrow==25.0.0 \
    pynwb==4.1.0 \
    scipy==1.18.0
