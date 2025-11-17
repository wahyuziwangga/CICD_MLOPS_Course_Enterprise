from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from app.model_loader import load_model

app = FastAPI(title="Enterprise MLOps API", version="1.0.0")

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: int
    probabilities: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 4:
        # Contoh bug yang sering muncul: return 500 → ini nanti kamu bahas di skenario hotfix
        raise HTTPException(status_code=400, detail="features must have length 4")
    #print("[HOTFIX DEMO] predict() endpoint called")
    model = load_model()
    X = np.array(req.features).reshape(1, -1)
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0].tolist()

    return PredictResponse(prediction=int(pred), probabilities=probs)
