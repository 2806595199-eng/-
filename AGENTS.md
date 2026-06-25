# 项目上下文

本文件用于承接 Claude Code / Codex 的项目长期上下文。后续如果有新的架构决策、现场约束、接口约定或已知问题，优先更新这里，让新的会话能快速接上。

## 项目目标

本项目是“深度除氟预测与加药推荐服务”。核心目标是基于水质记录、时序特征和模型预测，给出出水氟浓度预测、风险等级和 PACL / 除氟剂加药推荐。

当前版本从 README 和服务代码看是 `0.4.0`。主要服务形态是 FastAPI API，模型链路以 TabPFN 为主，XGBoost 作为备选/快速优化模型。

## 主要目录

- `core/`: 全局配置、字段定义、特征工程、TabPFN 运行时相关公共能力。
- `training/`: 数据加载、清洗、质量检查、训练、模型导出、模型注册、反馈增量更新和定时更新。
- `serving/`: FastAPI 服务、推理引擎、加药优化器、成本计算、泵流量换算、在线反馈历史。
- `tests/`: pytest 测试，覆盖 API、配置、数据清洗、特征工程、训练链路、优化器、模型注册和在线反馈。
- `scripts/`: 模型更新和定时更新脚本。
- `models/`: 训练/导出的模型产物目录。
- `logs/`: 服务运行日志和推荐/预测结果日志。
- `文档/`: 项目资料和数据文件。
- 根目录 `serve.py`、`train.py`、`gen_sim_data.py`: 兼容性 CLI 入口，实际逻辑在对应 package 内。

## 常用命令

本地安装依赖：

```bash
pip install -r requirements.txt
```

运行测试：

```bash
python -m pytest tests/ -v
```

启动 API：

```bash
python serve.py
```

训练模型：

```bash
python train.py --data path/to/data.csv --device cpu --output-dir models
```

Docker 部署：

```bash
cp .env.example .env
docker compose up -d --build
curl http://localhost:8000/api/v1/health
```

## API 入口

公开/主要接口包括：

- `GET /api/v1/health`: 服务存活检查。
- `GET /api/v1/ready`: 模型就绪检查。
- `POST /api/v1/dose/recommend/batch`: 批量水质记录加药推荐，支持同步和异步。
- `GET /api/v1/task/{task_id}`: 查询异步任务结果。

代码中还保留了若干隐藏接口用于预测、单次推荐、反馈记录和从反馈触发模型更新。

## 关键配置

主要配置在 `core/config.py` 和 `.env.example`：

- `MODEL_DEVICE`: `auto` / `cpu` / `cuda`。
- `USE_BACKUP`: 是否使用备选模型策略。
- `CPU_THREADS`: CPU 线程数，Docker 中会映射到常见数学库线程变量。
- `ALLOWED_ORIGINS`: CORS 白名单。
- `ONLINE_HISTORY_DIR`: 在线反馈/运行历史目录。
- `TABPFN_MODEL_CACHE_DIR`: TabPFN 缓存目录。

模型字段和控制参数集中在 `core/config.py`，包括 `MODEL_INPUT_COLS`、`TARGET_COL`、PACL/除氟剂范围、成本参数、HRT 延迟、lag/rolling 特征等。修改这些参数时要同步考虑训练、推理和测试。

## 开发注意事项

- 根目录入口文件主要用于兼容旧导入，新增业务逻辑应放在 `core/`、`training/` 或 `serving/`。
- `effluent_f` 是预测目标，不能作为模型输入特征；相关约束在 `core/config.py` 和测试里已有覆盖。
- 时序数据默认按时间前 80% 训练、后 20% 测试，不应随机打乱。
- 当前特征工程使用按变量独立的 HRT 延迟；不要同时启用全局 `OUTPUT_DELAY_STEPS` 造成重复延迟。
- API 测试会用 fake engine / fake optimizer，避免依赖真实 TabPFN 模型。
- 运行服务可能写入 `logs/`、`logs/results/` 和在线历史目录。
- Windows 终端里部分中文注释可能显示为乱码；编辑文件时仍应保持 UTF-8。

## 测试策略

改动后优先运行相关测试；涉及公共配置、字段、特征工程、训练或 API 返回结构时，建议运行完整测试：

```bash
python -m pytest tests/ -v
```

如果只改文档，通常无需运行完整测试，但应确认文件内容能正常读取。

## Claude Code 上下文迁移区

从 Claude Code 迁移上下文时，建议追加到本节，保留“决策和原因”，避免粘贴冗长聊天记录。

可以按下面格式补充：

```markdown
### YYYY-MM-DD 决策标题

- 背景:
- 决策:
- 原因:
- 影响文件/模块:
- 后续事项:
```

## 当前待补充信息

- 真实生产数据路径、字段来源和采样间隔是否有最终确认版本。
- PACL、除氟剂、PAM、磁粉等成本/密度/浓度参数是否已由甲方确认。
- 现场 PLC 点位和 `MODEL_INPUT_COLS` 的映射是否已冻结。
- TabPFN 在目标部署机器上的 GPU/CPU 性能基准。
