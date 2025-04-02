# From Chaos to Clarity: Log-based Kernel Panic Root Cause Analysis for Large-Scale Cloud Services

## Abstract
Operating system (OS) kernel panics, which occur when the OS encounters unrecoverable fatal errors, pose significant challenges to the stability and reliability of ByteDance’s large-scale cloud services. While analyzing kernel panic logs is crucial for diagnosing root
causes and preventing future occurrences, kernel panic root cause analysis (RCA) faces two major challenges: (1) the scarcity of logs that directly indicate kernel panics, and (2) the presence of complex long-range dependencies within kernel panic logs that complicate root cause identification. To address these challenges, we propose
LogSage, a novel log-based framework for kernel panic RCA in
large-scale cloud services. LogSage employs a hybrid approach that combines clustering techniques and large language models (LLMs) to effectively extract fault-indicating logs, while leveraging Graph
Neural Networks (GNNs) to capture and model the intricate relationships within these logs. We evaluated LogSage using a comprehensive dataset of 20,000 kernel panics from ByteDance’s production environment. The results demonstrate that LogSage significantly outperforms existing methods in both accuracy and robustness of root cause identification. Moreover, a six-month deployment of
LogSage in ByteDance’s cloud infrastructure has proven its practical value, successfully assisting engineers in real-world kernel panic RCA tasks, as validated through multiple case studies.

## System Architecture
graph TD
A[Log Input] --> B[Preprocessing]
B --> C[Graph Construction]
C --> D[GNN Model Training]
D --> E[Model Serving]
E --> F[API Interface]
F --> G[Frontend Display]
C --> H[Graph Storage]
D --> H

## Quick Start
cd LogSage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
# Install dependencies
pip install -r requirements.txt

## Data Process
python scripts/preprocess.py --input data/raw/ --output data/processed/

## Model Train
# Train GraphSAGE model
python scripts/train_model.py --model graphsage --epochs 50 --gpu

# Train GAT model
python scripts/train_model.py --model gat --epochs 50 --gpu

# Train GCN model
python scripts/train_model.py --model gcn --epochs 50 --gpu

# Start API service
python src/serving/api_server.py

# Or use in production environment
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.serving.api_server:app

## Project Structure
LogSage/
├── configs/               # Configuration files
│   ├── model_config.yaml
│   └── server_config.yaml
├── data/                  # Data directory
│   ├── raw/               # Raw logs
│   ├── processed/         # Processed data
│   └── streaming/         # Streaming log input
├── docs/                  # Documentation
├── models/                # Trained models
├── scripts/               # Utility scripts
│   ├── preprocess.py
│   ├── train_model.py
│   └── start_server.sh
├── src/                   # Source code
│   ├── KPRCA_stage/       # Core algorithms
│   │   ├── models/        # Model definitions
│   │   ├── trainer.py     # Training logic
│   │   └── graph_constructor.py
│   └── serving/           # Server-side code
│       ├── api_server.py
│       └── graph_updater.py
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md              # This document



## Development Guide

## Adding New Models
1. Create a new model file in src/KPRCA_stage/models/
2. Implement the base model interface
3. Update trainer.py to support the new model
4. Add test cases

## Contributing
1. Fork the repository
2. Create your feature branch (git checkout -b feature AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request


## License
Apache License 2.0

## Citation
If you use LogSage in your research, please cite

## Contact
For technical inquiries:
Email: [cuitianyu@mail.nankai.edu.cn]