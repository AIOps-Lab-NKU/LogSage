#!/bin/bash
# ./train_model.sh [--model MODEL_TYPE] [--epochs EPOCHS] [--gpu]

```bash
MODEL="graphsage"  # Options: graphsage, gat, gcn
EPOCHS=100
USE_GPU=false
CONFIG_FILE="../configs/training_config.yaml"
LOG_DIR="../logs/training"
DATA_DIR="../data/processed"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${MODEL}_${TIMESTAMP}.log"

DEVICE="cpu"
if [ "$USE_GPU" = true ] && nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "Detected GPU, CUDA acceleration will be used"
fi

echo "Starting training ${MODEL} model, epochs: ${EPOCHS}, device: ${DEVICE}"
echo "Training log: ${LOG_FILE}"

python ../src/train.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --config "$CONFIG_FILE" \
    --data-dir "$DATA_DIR" > "$LOG_FILE" 2>&1

# Check training results
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Best model saved to: ../models/best_${MODEL}.pth"
else
    echo "Training failed, please check the log file: $LOG_FILE"
    exit 1
fi
```