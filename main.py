"""
NetShield External Validation API
==================================
Run:
    pip install fastapi uvicorn
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from preprocessfunc import load_model, preprocess_window, predict_window, is_loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model("netshield_model")
    yield

app = FastAPI(
    title="NetShield Validation API",
    version="1.0.0",
    lifespan=lifespan,
)


class PacketFeatures(BaseModel):
    ip_protocol: int
    ip_dscp: int
    ip_ttl: int
    ip_flags_df: int
    ip_flags_mf: int
    dst_port: int
    tcp_syn: int
    tcp_ack: int
    tcp_fin: int
    tcp_rst: int
    tcp_psh: int
    icmp_type: int
    http_method: str
    http_response_code: int
    http_content_length: int
    http_content_type: str
    dns_query_type: str


class WindowPayload(BaseModel):
    window_id: int
    total_packets: int
    window_seconds: float
    features: list[PacketFeatures]


class ValidationResponse(BaseModel):
    is_attack: bool


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": is_loaded()}


@app.post("/validate", response_model=ValidationResponse)
def validate(payload: WindowPayload):
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    raw    = payload.model_dump()
    df     = preprocess_window(raw)
    result = predict_window(df)

    return ValidationResponse(is_attack=result["is_attack"])