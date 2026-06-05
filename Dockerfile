# FREEZE (reproducible builds): pin by digest at freeze time —
#   docker pull python:3.12-slim && docker inspect --format '{{index .RepoDigests 0}}' python:3.12-slim
# then replace the line below with:  FROM python@sha256:<digest>   (see FREEZE.md step 3)
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY backend/ ./backend/

EXPOSE 8001

# --workers 1 is REQUIRED: the async Council job store (council_async._JOBS) is in-process.
# More than one worker silently splits the store across processes (a poll may miss its job).
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
