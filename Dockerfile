FROM python:3.11-slim

# build tools (some wheels need them) -- done as root, before switching user
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
# CPU-only torch keeps the image small (no CUDA needed)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Build the vector index and download both models at BUILD time, so startup is fast
RUN python -m app.ingest \
    && python -c "from app.models import get_reranker; get_reranker()"

EXPOSE 7860
CMD ["bash", "start.sh"]
