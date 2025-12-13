FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY run.sh .
COPY Procfile .
COPY src/ ./src/

RUN mkdir -p logs runs
RUN chmod +x run.sh

EXPOSE 8010

CMD ["agentbeats", "run_ctrl"]
