import os
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.agent import OmniCartAgent
from engine.models import RecommendationResponse
from evals.run_evals import run_evaluation_benchmark

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "data", "products_catalog.json")
POLICIES_PATH = os.path.join(BASE_DIR, "data", "store_policies.json")
EVAL_PATH = os.path.join(BASE_DIR, "evals", "benchmark_report.json")
STATIC_DIR = os.path.join(BASE_DIR, "web")

app = FastAPI(
    title="OmniCart AI — E-Commerce Hybrid RAG & Regimen Agent",
    version="1.0.0",
    description="Production-grade e-commerce RAG agent with hybrid search, cross-encoder reranking, and deterministic guardrails."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent
agent = OmniCartAgent(CATALOG_PATH, POLICIES_PATH)

class ChatQueryRequest(BaseModel):
    query: str

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "OmniCart AI",
        "reranker": "FlashRank ms-marco-TinyBERT-L-2-v2",
        "catalog_size": len(agent.products),
        "policy_count": len(agent.policies)
    }

@app.post("/api/chat", response_model=RecommendationResponse)
def chat_recommendation(payload: ChatQueryRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return agent.process_query(payload.query.strip())

@app.get("/api/catalog")
def get_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/policies")
def get_policies():
    with open(POLICIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/evals")
def get_eval_results():
    if not os.path.exists(EVAL_PATH):
        # Run evals if report doesn't exist yet
        return run_evaluation_benchmark(output_report_path=EVAL_PATH)
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/evals/run")
def trigger_eval_run():
    return run_evaluation_benchmark(output_report_path=EVAL_PATH)

# Serve Web Frontend
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
