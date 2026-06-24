"""在线服务 — 深度除氟预测与加药推荐  v0.4.0

API:
    GET  /api/v1/health                健康检查
    GET  /api/v1/ready                 模型就绪检查
    POST /api/v1/dose/recommend/batch  批量上传水质，返回下一时刻加药推荐
    GET  /api/v1/task/{task_id}        异步任务结果查询
    POST /api/v1/feedback              上报实测出水氟（反馈闭环）
"""

import sys, os, time, argparse, json, inspect, uuid, threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from serving.inference_engine import InferenceEngine
from serving.optimizer import GridSearchDosingOptimizer
from training.model_registry import missing_artifacts
from training.update_scheduler import run_scheduled_update
from serving.online_history import (history_status, record_feedback,
                                    record_runtime_event)
from core import config as cfg

# ── 初始化 ──
engine = InferenceEngine("models")
try:
    engine.load()
    model_loaded = True
except Exception as e:
    print(f"Warning: model not loaded ({e})")
    model_loaded = False

optimizer = GridSearchDosingOptimizer()
start_time = time.time()
ONLINE_HISTORY_DIR = os.environ.get("ONLINE_HISTORY_DIR", "data/online")

_ALLOWED = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
_allowed_origins = [x.strip() for x in _ALLOWED.split(",") if x.strip()]

app = FastAPI(title="深度除氟预测与加药推荐服务", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins,
                   allow_methods=["*"], allow_headers=["*"])

from utils.logger import get_logger
logger = get_logger("serve")

_task_store: dict[str, dict] = {}
_task_lock = threading.Lock()

# ── 频率限制 ──
_last_call: dict[str, float] = {}
_RATE_LIMIT_SEC = 30  # 同一 IP 最短调用间隔


def _check_rate_limit(client_ip: str):
    """频率限制"""
    now = time.time()
    last = _last_call.get(client_ip, 0)
    if now - last < _RATE_LIMIT_SEC:
        raise HTTPException(status_code=429, detail=f"请求过于频繁，请间隔至少 {_RATE_LIMIT_SEC} 秒")
    _last_call[client_ip] = now


def _compute_risk(predicted_f: float) -> str:
    """基于预测出水氟判定风险等级"""
    if predicted_f <= 0.7:
        return "safe"
    elif predicted_f <= 0.9:
        return "warning"
    else:
        return "danger"


# ── 结果存档 ──
def _save_result(kind: str, data: dict, rec: dict = None):
    os.makedirs("logs/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if kind == "predict":
        record = {
            "时间": datetime.now().strftime("%m-%d %H:%M:%S"),
            "类型": "预测出水氟",
            "预测出水氟_mgL": data.get("predicted_f"),
            "最差_q95": data.get("q95"),
            "风险等级": data.get("risk_level"),
            "使用模型": data.get("model_used"),
        }
    else:
        record = {
            "时间": datetime.now().strftime("%m-%d %H:%M:%S"),
            "类型": "加药推荐",
            "推荐方案": {
                "PACl_mgL": rec["pacl_dose_setpoint"],
                "除氟剂_mLL": rec["defluor_dose_setpoint"],
                "预测出水氟": rec["predicted_f"],
                "风险等级": rec["risk_level"],
                "吨水成本_元": rec["cost_per_ton"],
            },
        }
    filepath = f"logs/results/{kind}_{ts}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


# ── Pydantic schemas ──

class WQInput(BaseModel):
    influent_flow: float = Field(..., gt=0, le=500, description="入水口流量 m³/h")
    influent_ph: float = Field(..., ge=0, le=14, description="入水PH值")
    conductivity: float = Field(..., ge=0, le=20000, description="入水电导率 μS/cm")
    influent_f: float = Field(..., ge=0, le=50, description="入水氟化物浓度 mg/L")
    pacl_dose: float = Field(..., ge=0, le=3000, description="当前PAC投加量 mg/L")
    defluor_dose: float = Field(..., ge=0, le=10, description="当前除氟剂投加量 mL/L")
    pacl_tank_ph: float = Field(7.0, ge=0, le=14)
    defluor_tank_ph: float = Field(6.0, ge=0, le=14)
    recycle_flow: float = Field(0.0, ge=0, le=10)
    waste_flow: float = Field(0.0, ge=0, le=10)
    pam_dose: float = Field(0.0, ge=0, le=10)
    timestamp: Optional[str] = None
    history: Optional[list[dict]] = None


class DoseReq(BaseModel):
    water_quality: WQInput
    mode: str = "balanced"
    priority: Optional[str] = None
    schemes: Optional[list[dict]] = None


class PredictResp(BaseModel):
    predicted_f: float
    q05: float
    q95: float
    risk_level: str
    model_used: str = "tabpfn"
    warnings: list = []
    record_id: Optional[str] = None


class DoseResp(BaseModel):
    recommended_mode: str
    recommended: dict
    alternatives: dict
    pareto_front: list
    risk_before: str = "unknown"
    risk_after: str = "unknown"
    warnings: list = []
    assumptions: dict = {}
    record_id: Optional[str] = None


class FeedbackInput(BaseModel):
    request_id: Optional[str] = Field(None, description="关联的异步任务 request_id")
    record_id: Optional[str] = Field(None, description="关联的推荐 record_id（同步模式）")
    effluent_f: float = Field(..., ge=0, le=10, description="实测出水氟浓度 mg/L")
    timestamp: Optional[str] = Field(None, description="采样时间 ISO 8601")
    executed_pacl_dose: Optional[float] = Field(None, ge=0, description="实际执行的PAC投加量 mg/L")
    executed_defluor_dose: Optional[float] = Field(None, ge=0, description="实际执行的除氟剂投加量 mL/L")


class FeedbackUpdateReq(BaseModel):
    min_rows: int = Field(100, ge=1)
    publish: bool = False
    min_r2: Optional[float] = None
    device: str = cfg.DEVICE


class BatchReq(BaseModel):
    records: list[dict] = Field(..., min_length=1, max_length=100,
        description="水质记录列表（最早→最新）， 1-100 条")
    mode: str = Field("balanced", description="safe / economic / balanced")
    timeout: int = Field(10, ge=1, le=30, description="超时秒数，默认10，最大30")
    async_mode: bool = Field(False, description="true 时异步返回 request_id，通过 GET /task/{id} 查结果")


# ── Routes ──

def _run_optimizer(wq: dict, mode: str, history=None):
    try:
        params = inspect.signature(optimizer.optimize).parameters
    except (TypeError, ValueError):
        params = {}
    if "history" in params:
        return optimizer.optimize(wq, engine, mode=mode, history=history)
    return optimizer.optimize(wq, engine, mode=mode)


def _readiness_payload() -> tuple[dict, bool]:
    models_root = getattr(engine, "models_root", "models")
    model_dir = getattr(engine, "models_dir", models_root)
    missing = missing_artifacts(model_dir)
    checks = {
        "model_loaded": bool(model_loaded),
        "main_model_fitted": bool(getattr(engine, "main_model_fitted", False)),
        "backup_model_loaded": getattr(engine, "backup_model", None) is not None,
        "missing_artifacts": missing,
    }
    ready = (
        checks["model_loaded"]
        and (checks["main_model_fitted"] or checks["backup_model_loaded"])
        and not missing
    )
    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "model_dir": str(model_dir),
    }, ready


def _do_batch_compute(records: list, mode: str) -> dict:
    """执行批量计算，返回结果字典"""
    import pandas as pd
    history = pd.DataFrame(records)
    current = history.iloc[-1].to_dict()
    for col in ("recycle_flow", "waste_flow", "pam_dose"):
        if col not in current:
            current[col] = 0.0
    t0 = time.time()
    opt = _run_optimizer(current, mode=mode, history=history)
    elapsed = round(time.time() - t0, 3)
    rec = opt["recommended"]
    # 使用优化器内 TabPFN 验证后的 q95 风险判定，而非基于均值的二次计算
    cur_pacl = current.get("pacl_dose", rec["pacl_dose_setpoint"])
    cur_deflu = current.get("defluor_dose", rec["defluor_dose_setpoint"])
    pacl_jump = abs(rec["pacl_dose_setpoint"] - cur_pacl) / max(cur_pacl, 1e-6)
    deflu_jump = abs(rec["defluor_dose_setpoint"] - cur_deflu) / max(cur_deflu, 1e-6)
    warnings = rec.get("warnings", [])[:]
    if pacl_jump > 0.5 or deflu_jump > 0.5:
        warnings.append(
            f"剂量跳变较大（PACl: {pacl_jump:.0%}, 除氟剂: {deflu_jump:.0%}），建议确认执行")
    if rec.get("risk_level") == "danger":
        warnings.append("所有候选方案均存在超标风险，建议人工介入")

    return {
        "pacl_dose_setpoint": rec["pacl_dose_setpoint"],
        "defluor_dose_setpoint": rec["defluor_dose_setpoint"],
        "predicted_f": rec["predicted_f"],
        "risk_level": rec["risk_level"],
        "effluent_limit": cfg.LIMIT_F,
        "safety_margin": round(cfg.LIMIT_F - rec["predicted_f"], 3),
        "mode": mode,
        "based_on_records": len(records),
        "elapsed_s": elapsed,
        "warnings": warnings,
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "version": "0.4.0",
        "model_version": engine.version,
        "model_loaded": model_loaded,
        "uptime": round(time.time() - start_time, 1),
    }


@app.get("/api/v1/ready")
def ready():
    payload, is_ready = _readiness_payload()
    if not is_ready:
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.get("/api/v1/history/status", include_in_schema=False)
def online_history_status():
    return history_status(ONLINE_HISTORY_DIR)


@app.post("/api/v1/feedback", include_in_schema=False)
def feedback(payload: FeedbackInput, request: Request):
    recorded = record_feedback(
        payload.model_dump(exclude_none=True),
        history_dir=ONLINE_HISTORY_DIR,
    )
    return {
        "status": "recorded",
        "feedback_id": recorded["feedback_id"],
        "record_id": recorded.get("record_id"),
    }


@app.post("/api/v1/model/update/from-feedback", include_in_schema=False)
def update_model_from_feedback(req: FeedbackUpdateReq):
    return run_scheduled_update(
        history_dir=ONLINE_HISTORY_DIR,
        models_root="models",
        min_rows=req.min_rows,
        auto_publish=req.publish,
        min_r2=req.min_r2,
        device=req.device,
    )


@app.post("/api/v1/predict", response_model=PredictResp, include_in_schema=False)
def predict(wq: WQInput):
    history = getattr(wq, "history", None)
    if history:
        import pandas as pd
        history = pd.DataFrame([h.model_dump() if hasattr(h, "model_dump") else h for h in history])
    result = engine.predict(wq.model_dump(), history=history)
    event = record_runtime_event("predict", wq.model_dump(), result, history_dir=ONLINE_HISTORY_DIR)
    result["record_id"] = event["record_id"]
    _save_result("predict", result)
    logger.info("predict", extra={"predicted_f": result["predicted_f"],
                "risk": result["risk_level"], "model": result["model_used"]})
    return PredictResp(**result)


@app.post("/api/v1/dose/recommend", response_model=DoseResp, include_in_schema=False)
def dose_recommend(req: DoseReq):
    wq = req.water_quality.model_dump()
    mode = req.priority if req.priority in ("economic", "balanced", "safe") else req.mode
    history = getattr(req.water_quality, "history", None)
    if history:
        import pandas as pd
        history = pd.DataFrame([h.model_dump() if hasattr(h, "model_dump") else h for h in history])
    pred_before = engine.predict(wq, history=history)
    opt = _run_optimizer(wq, mode=mode, history=history)
    rec = opt["recommended"]
    executed_wq = {**wq, "pacl_dose": rec["pacl_dose_setpoint"], "defluor_dose": rec["defluor_dose_setpoint"]}
    event = record_runtime_event("recommend", executed_wq, {
        "prediction_before": pred_before, "original_water_quality": wq,
        "recommended": rec, "mode": mode,
    }, history_dir=ONLINE_HISTORY_DIR)
    _save_result("recommend", {k: v for k, v in opt.items() if k != "pareto_front"}, rec=rec)
    logger.info("recommend", extra={"mode": mode, "pacl": rec["pacl_dose_setpoint"],
                "defluor": rec["defluor_dose_setpoint"], "cost": rec["cost_per_ton"], "risk": rec["risk_level"]})
    return DoseResp(recommended_mode=mode, recommended=rec, alternatives=opt["alternatives"],
                    pareto_front=opt["pareto_front"], risk_before=pred_before["risk_level"],
                    risk_after=rec["risk_level"], warnings=opt["warnings"],
                    assumptions=opt["assumptions"], record_id=event["record_id"])


@app.post("/api/v1/dose/recommend/batch")
def dose_recommend_batch(req: BatchReq, request: Request):
    _check_rate_limit(request.client.host if request.client else "unknown")

    if req.async_mode:
        task_id = uuid.uuid4().hex[:12]
        with _task_lock:
            _task_store[task_id] = {"status": "pending", "created_at": datetime.now(timezone.utc).isoformat()}
        # 后台线程执行计算
        def _run():
            try:
                result = _do_batch_compute(req.records, req.mode)
                with _task_lock:
                    _task_store[task_id] = {"status": "completed", "result": result}
            except Exception as e:
                with _task_lock:
                    _task_store[task_id] = {"status": "failed", "error": str(e)[:500]}
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"request_id": task_id, "status": "pending",
                "poll_url": f"/api/v1/task/{task_id}"}

    # 同步模式：用线程池 + 超时实现抢占式终止
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_batch_compute, req.records, req.mode)
        try:
            result = future.result(timeout=req.timeout)
        except FutureTimeoutError:
            future.cancel()
            raise HTTPException(status_code=504,
                detail=f"计算超时（>{req.timeout}s），请使用备用策略或减少记录数")
        except Exception as e:
            logger.error(f"batch compute failed: {e}")
            raise HTTPException(status_code=500, detail="模型计算异常，请使用上次推荐值")

    result["record_id"] = uuid.uuid4().hex[:16]
    return result


@app.get("/api/v1/task/{task_id}")
def get_task(task_id: str, request: Request):
    with _task_lock:
        task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task_id 不存在或已过期")
    return task


# ── 异常处理 ──

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={
        "error": "validation_error",
        "detail": str(exc.errors()),
        "hint": "请检查传感器数据是否在合理范围内",
    })


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={
        "error": exc.detail,
    })


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"unhandled {type(exc).__name__} at {request.url}: {str(exc)[:200]}")
    return JSONResponse(status_code=500, content={
        "error": "internal_server_error",
        "detail": "服务内部异常，请检查日志或联系运维",
    })


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
