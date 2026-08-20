# hash:sha256:98f6840d3fe21807848c3af5c486ab5dc7b531b374c091d9ea1b7fcc428f15e8
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

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
