FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY council_mcp_server.py .

EXPOSE 8002
ENV COUNCIL_BASE_URL=http://llm-council:8001 \
    COUNCIL_POLL_INTERVAL=3
# COUNCIL_MCP_TOKEN is injected at runtime from /opt/ecosystem/.env — NEVER baked in.

CMD ["uvicorn", "council_mcp_server:app", "--host", "0.0.0.0", "--port", "8002"]
