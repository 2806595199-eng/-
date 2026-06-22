# 深度除氟预测与加药推荐服务 API 说明

**文档版本**: v0.4.0　｜　**服务版本**: 0.4.0　｜　**更新日期**: 2025-06-16

**Base URL**: `http://<服务器IP>:8000`

---

## 变更日志

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v0.4.0 | 2025-06-16 | 新增异步模式、ISO 8601 时间戳、错误码表、频率限制、risk_level 阈值调整、effluent_limit/safety_margin 字段 |
| v0.3.0 | 2025-06-10 | 新增 batch 批量接口、简化 API 面、中试数据对齐 |
| v0.2.0 | 2025-05-28 | 初始版本 |

---

## 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 服务存活检查 |
| GET | `/api/v1/ready` | 模型就绪检查 |
| POST | `/api/v1/dose/recommend/batch` | **主接口**——上传历史水质，返回下一时刻加药推荐 |
| GET | `/api/v1/task/{request_id}` | 异步任务结果查询 |

---

## 频率限制

- 同一 IP 最短调用间隔：**30 秒**
- 单批次最大记录数：**100 条**
- 超频返回 `429 Too Many Requests`

---

## 1. 健康检查

```
GET /api/v1/health
```

**响应示例**：
```json
{
  "status": "ok",
  "version": "0.4.0",
  "model_version": "20260529_000012_120382",
  "model_loaded": true,
  "uptime": 3600.0
}
```

---

## 2. 就绪检查

```
GET /api/v1/ready
```

就绪返回 200，未就绪返回 503。Docker / K8s 健康探测用。

---

## 3. 加药推荐（主接口）

**语义**：输入 N 条历史水质记录（最早→最新），输出基于全部历史计算的**下一时刻**推荐加药量。

```
POST /api/v1/dose/recommend/batch
Content-Type: application/json
```

### 3.1 请求参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| records | array | ✅ | — | 水质记录列表（最早→最新），1-100 条 |
| mode | string | — | `"balanced"` | 策略：`safe` / `economic` / `balanced` |
| timeout | int | — | 10 | 同步模式超时秒数，1-30 |
| async_mode | bool | — | false | `true` 时立即返回 request_id，通过 GET /task/{id} 查结果 |

每条 record：

| # | 字段 | PLC地址 | 类型 | 必填 | 单位 | 范围 | 说明 |
|---|------|---------|------|------|------|------|------|
| — | timestamp | — | string | ✅ | ISO 8601 | — | 采样时间，格式 `2025-06-16T14:00:00+08:00` |
| 1 | influent_flow | DBD0 | float | ✅ | m³/h | 0-500 | 入水口流量（当前值） |
| 2 | influent_ph | DBD4 | float | ✅ | — | 0-14 | 入水 pH（当前值） |
| 3 | conductivity | DBD8 | float | ✅ | μS/cm | 0-20000 | 入水电导率（当前值） |
| 4 | influent_f | DBD12 | float | ✅ | mg/L | 0-50 | 入水氟化物浓度（当前值） |
| 5 | pacl_dose | DBD16 | float | ✅ | mg/L | 0-3000 | PAC 投加量（当前值） |
| 6 | defluor_dose | DBD20 | float | ✅ | mL/L | 0-10 | 除氟剂投加量（当前值） |
| 7 | pacl_tank_ph | DBD24 | float | ✅ | — | 0-14 | PACl 反应池 pH（当前值） |
| 8 | defluor_tank_ph | DBD28 | float | ✅ | — | 0-14 | 除氟反应池 pH（当前值） |
| 9 | recycle_flow | DBD32 | float | — | m³/h | 0-10 | 回流流量（当前值） |
| 10 | waste_flow | DBD36 | float | — | m³/h | 0-10 | 排泥流量，PLC 称"剩余流量"（当前值） |
| 11 | pam_dose | DBD40 | float | — | mg/L | 0-10 | PAM 投加量（当前值） |

### 3.2 请求示例

```json
{
  "records": [
    {
      "timestamp": "2025-06-16T14:00:00+08:00",
      "influent_flow": 101.4,
      "influent_ph": 7.12,
      "conductivity": 6556,
      "influent_f": 18.85,
      "pacl_dose": 775.4,
      "defluor_dose": 3.02,
      "pacl_tank_ph": 6.95,
      "defluor_tank_ph": 5.87,
      "recycle_flow": 0.32,
      "waste_flow": 2.42,
      "pam_dose": 3.85
    },
    {
      "timestamp": "2025-06-16T14:10:00+08:00",
      "influent_flow": 108.6,
      "influent_ph": 7.29,
      "conductivity": 6537,
      "influent_f": 18.62,
      "pacl_dose": 1107.7,
      "defluor_dose": 1.37,
      "pacl_tank_ph": 7.01,
      "defluor_tank_ph": 6.03,
      "recycle_flow": 0.30,
      "waste_flow": 2.45,
      "pam_dose": 3.92
    }
  ],
  "mode": "balanced",
  "timeout": 10,
  "async_mode": false
}
```

### 3.3 响应（同步模式）

```json
{
  "pacl_dose_setpoint": 1712.8,
  "defluor_dose_setpoint": 1.48,
  "predicted_f": 0.92,
  "risk_level": "danger",
  "effluent_limit": 1.0,
  "safety_margin": 0.08,
  "mode": "balanced",
  "based_on_records": 2,
  "elapsed_s": 0.03,
  "record_id": "4ca99fdb7a374bcd"
}
```

| 字段 | 类型 | 说明 | 单位 |
|------|------|------|------|
| pacl_dose_setpoint | float | PAC 推荐投加量（推荐值）→ 写入 PLC DBD72 | mg/L |
| defluor_dose_setpoint | float | 除氟剂推荐投加量（推荐值）→ 写入 PLC DBD68 | mL/L |
| predicted_f | float | 按推荐投加后预测出水氟浓度 | mg/L |
| risk_level | string | 风险等级，见下方阈值定义 | — |
| effluent_limit | float | 排放标准红线（固定 1.0） | mg/L |
| safety_margin | float | 安全余量 = effluent_limit - predicted_f | mg/L |
| mode | string | 本次使用的策略 | — |
| based_on_records | int | 基于几条历史记录计算 | — |
| elapsed_s | float | 计算耗时 | 秒 |
| record_id | string | 本次推荐唯一标识，反馈时使用 | — |

### 3.4 风险等级阈值

| risk_level | 条件 | PLC 处理建议 |
|------------|------|-------------|
| `safe` | predicted_f ≤ 0.7 | 正常执行推荐值 |
| `warning` | 0.7 < predicted_f ≤ 0.9 | 执行推荐值，关注趋势 |
| `danger` | predicted_f > 0.9 | 执行推荐值，检查设备/传感器 |

### 3.5 异步模式

`async_mode: true` 时立即返回：

```json
{
  "request_id": "a1b2c3d4e5f6",
  "status": "pending",
  "poll_url": "/api/v1/task/a1b2c3d4e5f6"
}
```

PLC 轮询 `GET /api/v1/task/{request_id}` 获取结果：

```json
{
  "status": "completed",
  "result": {
    "pacl_dose_setpoint": 1712.8,
    "defluor_dose_setpoint": 1.48,
    "...": "..."
  }
}
```

状态值: `pending`（计算中）→ `completed`（完成） / `failed`（失败）

### 3.6 错误响应

| 状态码 | 场景 | PLC 应如何处理 |
|--------|------|---------------|
| 200 | 正常 | 使用推荐值 |
| 400 | 请求参数错误（格式、必填缺失） | 检查 JSON 格式后重试 |
| 404 | 异步 task_id 不存在 | 确认 request_id 正确 |
| 422 | 数据校验失败（如 pH>14） | 检查传感器读数 |
| 429 | 调用过于频繁 | 等待 30 秒后重试 |
| 500 | 模型内部计算异常 | 使用上次推荐值，告警 |
| 503 | 模型未加载/服务未就绪 | 等待后重试，联系运维 |
| 504 | 计算超时（>timeout 秒） | 使用上次推荐值，下次减少记录数 |

---

## 4. 异步任务查询

```
GET /api/v1/task/{request_id}
```

### 4.1 计算中

```json
{
  "status": "pending",
  "created_at": "2025-06-16T06:30:00Z"
}
```

### 4.2 计算完成

```json
{
  "status": "completed",
  "result": {
    "pacl_dose_setpoint": 1712.8,
    "...": "..."
  }
}
```

### 4.3 计算失败

```json
{
  "status": "failed",
  "error": "错误详情"
}
```

---

## 5. 模式说明

| mode | 策略 | 适用场景 |
|------|------|----------|
| `safe` | 优先出水达标（predicted_f 尽可能低），药剂用量保守 | 进水氟波动大、排放检查严格时 |
| `economic` | 达标前提下药剂成本最低 | 进水稳定、成本敏感时 |
| `balanced` | 综合平衡成本与水质安全 | 默认推荐，日常运行 |

---

## 数据来源依据

| # | 字段 | 代码来源 | 中试完整率 | 上位机 PLC |
|---|------|----------|-----------|-----------|
| 1 | influent_flow | MODEL_INPUT_COLS | 40/40 ✓ | DB19.DBD0 ✓ |
| 2 | influent_ph | MODEL_INPUT_COLS | 39/40 ✓ | DB19.DBD4 ✓ |
| 3 | conductivity | MODEL_INPUT_COLS | 34/40 ✓ | DB19.DBD8 ✓ |
| 4 | influent_f | MODEL_INPUT_COLS | 40/40 ✓ | DB19.DBD12 ✓ |
| 5 | pacl_dose | MODEL_INPUT_COLS | 38/40 ✓ | DB19.DBD16 ✓ |
| 6 | defluor_dose | MODEL_INPUT_COLS | 35/40 ✓ | DB19.DBD20 ✓ |
| 7 | pacl_tank_ph | MODEL_INPUT_COLS | 40/40 ✓ | DB19.DBD24 ✓ |
| 8 | defluor_tank_ph | MODEL_INPUT_COLS | 40/40 ✓ | DB19.DBD28 ✓ |
| 9 | recycle_flow | MODEL_INPUT_COLS | 19/40 △ | DB19.DBD32 ✓ |
| 10 | waste_flow | MODEL_INPUT_COLS | 16/40 △ | DB19.DBD36 ✓ |
| 11 | pam_dose | MODEL_INPUT_COLS | 32/40 ✓ | DB19.DBD40 ✓ |
| — | effluent_f | TARGET_COL | 40/40 ✓ | DB19.DBD44 ✓ |

> △ 标记字段中试阶段覆盖率不足，后续中试数据将补齐。当前 API 在缺失时自动补 0 兼容。

---

## 调用流程

```
┌──────────┐     ┌──────────────────┐     ┌──────────┐
│  PLC      │     │  AI API          │     │  计量泵    │
│  DB19     │     │  :8000           │     │          │
└────┬─────┘     └───────┬──────────┘     └────┬─────┘
     │                   │                     │
     │ ① 每10分钟采集     │                     │
     │   11个寄存器值      │                     │
     │                   │                     │
     │ ② POST /batch     │                     │
     │   3-6条历史记录    │                     │
     │ ─────────────────→│                     │
     │                   │ ③ 特征工程+优化      │
     │                   │   网格搜索1600组合    │
     │                   │                     │
     │ ④ 返回推荐值       │                     │
     │ ←─────────────────│                     │
     │                   │                     │
     │ ⑤ 写入 DBD72 PAC  │                     │
     │   写入 DBD68 除氟剂│ ─────────────────→  │
     │                   │                     │ ⑥ 调节泵
     │                   │                     │
```

---

## 响应时间 SLA

| 指标 | 目标值 | 超时处理 |
|------|--------|----------|
| p99 同步响应 | ≤ 5 秒 | 超时返回 504，PLC 使用上次推荐值 |
| p50 同步响应 | ≤ 3 秒 | — |
| 异步模式 | 立即返回 request_id | 计算完成后 PLC 主动轮询 |
