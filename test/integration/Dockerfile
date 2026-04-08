FROM ceph-build:main.ubuntu22.04

# Proxy support (parameterized, NOT hardcoded)
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

# Install additional deps (most already in ceph-build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeigen3-dev curl python3 cmake build-essential git \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Copy sys/ to /ceph (preserves CMakeCache.txt path CMAKE_HOME_DIRECTORY=/ceph)
COPY sys/ /ceph/

# Copy Cortex sources under /cortex/
COPY hw/ /cortex/hw/
COPY driver/ /cortex/driver/
COPY include/ /cortex/include/

# Build hw/ mock .so
RUN mkdir -p /build/hw && cd /build/hw && \
    cmake /cortex/hw \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DFAISS_DOWNLOAD_URL_OVERRIDE="" && \
    make -j$(nproc) && \
    make install

# Build driver/ .so
RUN mkdir -p /build/driver && cd /build/driver && \
    cmake /cortex/driver \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCORTEX_HW_LIB_DIR=/usr/local/lib && \
    make -j$(nproc) && \
    make install

# Run ldconfig so loader finds the new .so files
RUN ldconfig

# Build the GTest integration test binary (not built by default)
RUN ninja -C /ceph/build.u2204 ceph_test_cortex_integration

# Build semantic_op_tool (may already exist)
RUN ninja -C /ceph/build.u2204 semantic_op_tool || true

# Copy fixture data
COPY hw/test/fixtures/ /cortex/hw/test/fixtures/

# Copy entrypoint
COPY test/integration/entrypoint.sh /cortex/test/integration/entrypoint.sh
RUN chmod +x /cortex/test/integration/entrypoint.sh

# Unset proxy for runtime (only needed during build)
ENV http_proxy=""
ENV https_proxy=""

WORKDIR /ceph
ENTRYPOINT ["/cortex/test/integration/entrypoint.sh"]
