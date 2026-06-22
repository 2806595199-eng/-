# 深度除氟预测与加药推荐服务

版本 0.4.0

## 快速部署（Docker）

```bash
cp .env.example .env
docker compose up -d --build
curl http://localhost:8000/api/v1/health
```

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 服务存活检查 |
| GET | `/api/v1/ready` | 模型就绪检查 |
| POST | `/api/v1/dose/recommend/batch` | 上传水质记录，返回加药推荐 |
| GET | `/api/v1/task/{request_id}` | 异步任务结果查询 |

Swagger 文档：`http://localhost:8000/docs`

## 配置

环境变量见 `.env.example`，关键项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_DEVICE | auto | cpu / cuda |
| USE_BACKUP | false | false=TabPFN 主模型；true=XGBoost 替代（CPU 应急） |
| CPU_THREADS | 4 | CPU 线程数 |
| ALLOWED_ORIGINS | localhost:3000 | CORS 白名单（逗号分隔） |

## 测试

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## 训练

```bash
python train.py --data 数据路径.csv [--device cpu]
```
