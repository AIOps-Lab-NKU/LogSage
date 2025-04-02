#!/bin/bash
# ./start_server.sh [--port PORT] [--env ENV_FILE]

```bash
PORT=8000
ENV_FILE="../configs/production.env"
LOG_DIR="../logs"
PID_FILE="../run/server.pid"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --env)
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$PID_FILE")"

if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Warning: Environment variable file not found: $ENV_FILE"
fi

if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

if ! pip show -q torch torch-geometric; then
    echo "Error: Missing required Python dependencies"
    exit 1
fi

echo "Starting LogSage service, port: $PORT"
nohup python ../src/server.py --port "$PORT" > "$LOG_DIR/server.log" 2>&1 &

echo $! > "$PID_FILE"
echo "Service started, PID: $(cat "$PID_FILE")"
echo "Log file: $LOG_DIR/server.log"
```