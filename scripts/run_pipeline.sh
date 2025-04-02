# scripts/run_pipeline.sh
#!/bin/bash

# 1. Data Preprocessing
python src/file_stage/clustering.py --input data/raw --output data/processed/clusters

# 2. LLM Explanation Generation
python src/file_stage/llm_explain.py --clusters data/processed/clusters --output data/processed/explanations

# 3. Graph Structure Construction
python src/kprca_stage/graph_constructor.py --features data/processed/tfidf_features.pkl --output data/graphs

# 4. Model Training
python src/kprca_stage/trainer.py --graph data/graphs/global_graph.pt --epochs 100

# 5. Start Service
python src/serving/api_server.py --model models/graphsage --graph data/graphs/global_graph.pt