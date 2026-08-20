# hash:sha256:e4919cba0093515b111e6777abc1b193a76e57905b2e6b0dd2808569ea6ba74a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/swdb-base-image:0.1

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
