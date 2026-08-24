# hash:sha256:7838798355476eb66820e1e6129ce791ab6bef9ed8d56acb7a7d6063aa988b0b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.129.0python3.12.4codeocean-mcp-serveruvpyreflyetc

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
