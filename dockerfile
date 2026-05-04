# nvidia-smi
# nvidia-container-toolkit --version
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Create working directory
ENV WORKDIR=/app
WORKDIR ${WORKDIR}

# Install system packages (git needed for GitPython; strace for debugging)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git \
        strace \
        apt-transport-https \
        ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install package
RUN pip install --upgrade pip
RUN pip install micm-nlp==0.1.0

# Copy project files
COPY . .

# Check if CUDA is available:
CMD [ "/bin/bash", "-c",  "python -c \"import torch; print(f'CUDA is available: {torch.cuda.is_available()}')\"" ]
