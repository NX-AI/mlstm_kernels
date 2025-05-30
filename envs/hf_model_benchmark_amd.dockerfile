FROM rocm/pytorch:rocm6.4_ubuntu24.04_py3.12_pytorch_release_2.6.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set workspace directory
WORKDIR /workspace

# Copy requirements into image
COPY ./environment_pt240_rocm64.txt ./requirements.txt

# Install Python requirements for benchmarking
RUN pip install --no-cache-dir -r requirements.txt

# Set default command to bash
CMD ["/bin/bash"]