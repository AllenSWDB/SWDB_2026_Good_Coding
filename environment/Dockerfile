# hash:sha256:019dc9cd02778e4dcb029a0c9f744eb249c54c79a02e349f0b5d816bd6a9a6fd
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.129.0python3.12.4codeocean-mcp-serveruvpyreflyetc

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    argparse \
    matplotlib==3.11.0 \
    numpy==2.5.1 \
    pandas==3.0.3 \
    pyarrow==25.0.0
