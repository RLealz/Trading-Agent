FROM python:3.11-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 agent
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir ".[live]"
USER agent
VOLUME ["/app/state"]
ENTRYPOINT ["trading-agent", "--config", "config/default.toml"]
CMD ["paper", "--live-data", "binance"]
