#
#   docker build -t aqwenator .
#   docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/home/aq/.cache/huggingface:ro \
#       aqwenator Qwen/Qwen3.5-9B
#
# The model must already be in the mounted Hugging Face cache (or pass a directory).
# Building needs no GPU. GPU_ARCH is the oldest architecture the kernels are compiled for;
# the driver compiles them for the device it finds at start-up.
# Base images are pinned by digest (multi-arch indexes of nvidia/cuda 13.0.3, Ubuntu 24.04).
ARG CUDA_DEVEL=nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
ARG CUDA_BASE=nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1

FROM ${CUDA_DEVEL} AS build
ARG RUST_TOOLCHAIN=1.99.1
# Must match the cuda-rust rev in Cargo.toml.
ARG CUDA_RUST_REV=8f5f2a5d6c9a5d16a75036fbcc639f751b68d537
ARG RUSTUP_VERSION=1.29.3
ARG GPU_ARCH=sm_80

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git gnupg build-essential pkg-config \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] https://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-21 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends clang-21 libclang-21-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LIBCLANG_PATH=/usr/lib/llvm-21/lib \
    PATH=/root/.cargo/bin:/usr/lib/llvm-21/bin:$PATH

# rustup-init at a fixed version, checked against its published SHA-256.
RUN case "$(dpkg --print-architecture)" in \
      arm64) target=aarch64-unknown-linux-gnu sha=9f608c142e0e38a78957d0661ba1dfaf0b6fb37184cfaa4b9a0aa57af6184282 ;; \
      *) target=x86_64-unknown-linux-gnu sha=b1c8cb14e30889e7e2999e6b69e62d9e9b9ffc0f4df8180564d306d9680e6c55 ;; \
    esac \
    && curl -sSf -o /tmp/rustup-init "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/${target}/rustup-init" \
    && echo "${sha}  /tmp/rustup-init" | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --default-toolchain none --profile minimal \
    && rm /tmp/rustup-init \
    && rustup toolchain install ${RUST_TOOLCHAIN} --profile minimal -c rust-src -c rustc-dev -c llvm-tools \
    && cargo +${RUST_TOOLCHAIN} install --git https://github.com/NVIDIA/cuda-rust.git --rev ${CUDA_RUST_REV} --locked cargo-oxide

WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/target \
    cargo oxide build --arch ${GPU_ARCH} -- --release --locked -p aq-server -p aq-engine --bins \
    && mkdir /out \
    && cp target/release/aq-serve target/release/aq-bench target/release/aq-check target/release/aq-ceiling /out/

# The binaries alone (`--target binaries --output type=local,dest=bin`); they need
# libcublasLt.so.13 and the NVIDIA driver on the host.
FROM scratch AS binaries
COPY --from=build /out/ /
COPY LICENSE /

FROM ${CUDA_BASE}
# cuBLASLt is the only toolkit library the engine links; libcuda comes from the host driver.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcublas-13-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /usr/local/bin/
COPY LICENSE /usr/share/doc/aqwenator/LICENSE
# Not root: the server only reads the model cache and listens on a high port.
RUN useradd --create-home --uid 10001 aq
USER aq
EXPOSE 8000
# aq-serve listens on localhost by default; in a container the port is published.
ENV AQ_HOST=0.0.0.0
ENTRYPOINT ["aq-serve"]
