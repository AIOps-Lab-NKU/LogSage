import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from KPRCA_stage.models import GraphSAGE, GAT, GCN
from .graph_updater import GraphUpdater

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LogSage API Server",
    description="A log analysis service based on graph neural networks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_updater = None
current_model = None
label_encoder = None

class LogRequest(BaseModel):
    log_lines: List[str]
    session_id: Optional[str] = None

class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    similar_logs: List[Dict]
    timestamp: str

class ModelInfo(BaseModel):
    model_type: str
    input_dim: int
    output_dim: int
    last_updated: str

@app.on_event("startup")
async def startup_event():
    """Initialize the model and graph at startup"""
    global graph_updater, current_model, label_encoder
    
    graph_updater = GraphUpdater(
        model_path=os.getenv("MODEL_PATH", "../models"),
        data_dir=os.getenv("DATA_DIR", "../data/streaming"),
        update_interval=int(os.getenv("UPDATE_INTERVAL", "3600"))
    )
    graph_updater.start()
    
    model_type = os.getenv("MODEL_TYPE", "graphsage")
    model_path = os.path.join(graph_updater.model_path, f"best_{model_type}.pth")
    
    try:
        if model_type == "graphsage":
            current_model = GraphSAGE.load_model(model_path, 1000, 64, 32)
        elif model_type == "gat":
            current_model = GAT.load_model(model_path, 1000, 8, 32)
        elif model_type == "gcn":
            current_model = GCN.load_model(model_path, 1000, 64, 32)
        
        with open(os.path.join(graph_updater.model_path, "label_encoder.json"), 'r') as f:
            label_encoder = json.load(f)
        
        logger.info(f"Loaded {model_type} model and label encoder")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    if graph_updater:
        graph_updater.stop()

@app.post("/predict", response_model=PredictionResult)
async def predict_log_anomaly(request: LogRequest):
    if not current_model or not graph_updater:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        processed_log = preprocess_log(request.log_lines)
        
        graph = graph_updater.get_current_graph()
        if graph is None:
            raise HTTPException(status_code=503, detail="Graph not initialized")
        
        prediction, confidence = make_prediction(processed_log, graph)
        
        similar_logs = find_similar_logs(processed_log, graph)
        
        return PredictionResult(
            prediction=prediction,
            confidence=float(confidence),
            similar_logs=similar_logs,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    if not current_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type=current_model.__class__.__name__,
        input_dim=current_model.conv1.in_channels,
        output_dim=current_model.conv2.out_channels,
        last_updated=datetime.fromtimestamp(
            os.path.getmtime(os.path.join(graph_updater.model_path, f"best_{current_model.__class__.__name__.lower()}.pth"))
        ).isoformat()
    )

def preprocess_log(log_lines: List[str]) -> str:
    return " ".join(log_lines)

def make_prediction(log_text: str, graph) -> tuple:
    return "Normal", 0.95

def find_similar_logs(log_text: str, graph, top_k: int = 5) -> List[Dict]:
    return [{
        "log": "sample log",
        "similarity": 0.8,
        "label": "Normal"
    }]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)