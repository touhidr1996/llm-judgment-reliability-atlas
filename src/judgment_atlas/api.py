from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .model import FEATURES

ROOT=Path(__file__).resolve().parents[2]
app=FastAPI(title="LLM Judgment Reliability Atlas API",version="0.1.0")


class JudgmentCase(BaseModel):
    context:str
    model:str
    persona:str
    intervention:str
    lexical_overlap:float=Field(ge=0,le=1)
    ambiguity:float=Field(ge=0,le=1)
    boundary_distance:float=Field(ge=0,le=1)


@lru_cache
def risk_model():
    path=ROOT/"artifacts"/"model.joblib"
    if not path.exists():raise FileNotFoundError("Run the pipeline first")
    return joblib.load(path)


@app.get("/health")
def health():return {"status":"ok","model_ready":(ROOT/"artifacts"/"model.joblib").exists()}


@app.post("/triage")
def triage(case:JudgmentCase):
    try:
        p=float(risk_model().predict_proba(pd.DataFrame([case.model_dump()])[FEATURES])[:,1][0])
        return {"disagreement_risk":p,"route":"deterministic_or_human_verification" if p>=.5 else "automated_with_logging","notice":"Synthetic research simulation; not a real LLM evaluation."}
    except FileNotFoundError as exc:raise HTTPException(status_code=503,detail=str(exc)) from exc
