# COMPASS Platform — Docker build for Railway (frontend + API)

# Stage 0: Build frontend static files (Node.js discarded after this stage)
FROM node:20-slim AS frontend
WORKDIR /ui
COPY compass-ui/package.json compass-ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY compass-ui/ ./
RUN npm run build

# Stage 1: Build Python packages that need compilers
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY compass/ ./compass/
# Install CPU-only PyTorch first (small ~200MB vs ~2GB for CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# disc covers scikit-learn + lightgbm; skip ml extra (umap-learn/numba not needed at runtime)
RUN pip install --no-cache-dir -e ".[primers,api,viz,disc]"
RUN pip install --no-cache-dir xgboost>=2.0

# RNA-FM package (structural embeddings for crRNA folding/accessibility)
RUN cd /tmp && git clone --depth 1 https://github.com/ml4bio/RNA-FM.git && \
    cd RNA-FM && touch README_backup.md && pip install --no-cache-dir -e . && \
    cd / && rm -rf /tmp/RNA-FM/.git

# RNA-FM weights: downloaded at first pipeline run from HuggingFace CDN (~30s)
COPY scripts/download_rnafm.py /app/scripts/download_rnafm.py

# Stage 2: Lean runtime (no compilers)
FROM python:3.11-slim
WORKDIR /app

# Only bowtie2 at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    bowtie2 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder (includes RNA-FM, xgboost, etc.)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy RNA-FM source from builder (editable install needs it)
COPY --from=builder /tmp/RNA-FM /tmp/RNA-FM

# Application code
COPY pyproject.toml README.md ./
COPY compass/ ./compass/
COPY api/ ./api/
COPY configs/ ./configs/
COPY data/ ./data/

# Frontend static files (built in Stage 0, ~5MB)
COPY --from=frontend /ui/dist ./compass-ui/dist/

# Compass-ML model package (architecture + checkpoints + features)
COPY compass-net/ ./compass-net/

# Editable install (egg-link only, no downloads)
RUN pip install --no-cache-dir --no-deps -e .

# Build Bowtie2 index
RUN bowtie2-build data/references/H37Rv.fasta data/references/H37Rv

RUN mkdir -p results/api results/panels results/validation

# Memory optimisation for constrained Railway containers
ENV MALLOC_TRIM_THRESHOLD_=0
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1

# PyTorch CPU threading — 2 threads per pipeline, 2 concurrent pipelines
# Total: 4 active threads fits Railway 8 vCPU shared infra
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV TORCH_NUM_THREADS=2

# Railway sets $PORT dynamically via env var
ENV PORT=8000
EXPOSE 8000
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
