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

### 2025-06-25 优化器架构定版

- 背景: 经过多次迭代（XGBoost粗筛→TabPFN全量→回退两阶段），最终确定了CPU/GPU自适应架构。
- 决策: **XGBoost 15维粗筛 + TabPFN 210维 batch校验15候选**。XGBoost 特征中剂量列使用候选当前值（非HRT延迟历史值），非剂量列使用HRT延迟历史值。TabPFN 校验池 = 模式三选(eco/bal/safe) + 约束池 top 15，去重后 batch predict 一次跑完。
- 原因: CPU上TabPFN单次推理43s，全量1600候选要4小时+。GPU上全量~2.2min可用但甲方只有CPU。XGBoost 0.2s扫1600个，TabPFN batch验15个~50s，10min采样间隔够用。
- 影响: `serving/optimizer.py` 验证池逻辑, `serving/inference_engine.py` predict_batch分路, `core/config.py` FAST_OPTIMIZER_MODEL=”xgboost”。
- 后续: GPU到货后可将210维统一+XGBoost替换为全量TabPFN。存量优化：历史特征预建一次+候选拼差异列可把GPU全量从2.2min降到5s。

### 2025-06-25 特征工程统一后回退

- 背景: 尝试让XGBoost也用210维特征（与TabPFN共享同一套HRT/lag/rolling），消除两模型特征不一致。
- 决策: **回退到15维**。训练侧XGBoost恢复独立FeatureEngineer（空lag/rolling），推理侧恢复`_build_simple_features`快速路径。
- 原因: CPU上XGBoost全量210维transform从0.2s涨到86s，慢400倍。15维虽无时序特征但足够区分候选排序，精确预测由TabPFN保证。
- 影响: `training/train.py` XGBoost训练路径, `serving/inference_engine.py` predict_batch XGBoost分支。

### 2025-06-24 甲方生产数据评估

- 背景: 收到甲方7天60万条生产数据（`api_records_2026-06-15_7d.json`），含全部11个输入+effluent_f。
- 决策: **数据不可用于训练。**
- 原因: (1) effluent_f仅122个不同值，97%连续行不变——传感器大概率故障；(2) 系统99.98%时间流量<1，仅106条运行记录；(3) 电导率67.5%时间报物理不可能的~2μS/cm；(4) effluent_f与投药量/进水氟无任何相关性。最小可用训练集：0条。
- 后续: 需甲方确认 effluent_f 传感器和 conductivity 传感器是否正常。要求至少100条以上有实际化验值配对的有效运行记录。

### 2025-06-22 PLC寄存器确认

- 背景: 甲方反馈加药程序更新完成，明确了AI需要提供的变量。
- 决策: API输出确认——`pacl_dose_setpoint`→DBD72, `defluor_dose_setpoint`→DBD68。新增模式开关`DBX76.0`（0=手动,1=AI控制），PLC侧负责。
- 影响: `core/config.py` 注释补充PLC地址映射。
- 后续: 无需改API代码，开关是PLC侧逻辑。

### 2025-06-22 API简化与feedback移除

- 背景: 多次迭代后确定甲方只需batch接口做定时批量数据上传和单次加药推荐。
- 决策: Swagger公开仅4接口(health/ready/batch/task)。feedback和model/update隐藏，dose/recommend/simple删除。`records`最小长度改为10条。异步模式保留(`async_mode=true`)。
- 原因: 甲方PLC定时采集→打包上传→获取推荐→执行，无需单条实时接口。反馈闭环后续通过工控机定时拉取数据实现。
- 影响: `serving/serve.py` 路由精简, API文档更新。

### 2025-06-16 特征工程保留但优化器分层

- 背景: 多次讨论TabPFN训练的意义——210维特征工程（HRT延迟+lag+rolling+派生）是否在优化器中发挥作用。
- 决策: **TabPFN校验时使用完整210维特征**，证明特征工程价值。XGBoost粗筛用15维简单特征即可区分候选排序。
- 原因: XGBoost只需相对排序（哪个候选更优），不需绝对精度。TabPFN做最终精度保障。

### 2025-06-14 项目中试数据对齐

- 背景: 综合三份资料（项目代码config.py、中试原始数据40批次、上位机PLC DB19寄存器表）交叉验证。
- 决策: 11个输入字段全部三方一致。`waste_flow` PLC中标注为”剩余流量”，确认为同一物理量。PAC为10%溶液(100g/L)，除氟剂为15%溶液。价格/浓度全部对齐中试原始数据。
- 影响: `core/config.py` 文档头更新、TODO精简。

## 当前待补充信息

- ~~真实生产数据路径、字段来源和采样间隔~~ → 甲方提供了7天60万条数据但不可用（见2025-06-24记录）
- PACL、除氟剂、PAM、磁粉等成本/密度/浓度参数：价格已对齐中试数据，密度1.4为估值待确认，Al质量分数/磁粉损耗率低优先级
- ~~现场 PLC 点位和 MODEL_INPUT_COLS 的映射~~ → 已冻结（见2025-06-22记录）
- TabPFN 在目标部署机器上的 GPU/CPU 性能基准: CPU 43s/次, GPU(RTX3060) 2s/次。甲方只有CPU。

## 当前待补充信息

- 真实生产数据路径、字段来源和采样间隔是否有最终确认版本。
- PACL、除氟剂、PAM、磁粉等成本/密度/浓度参数是否已由甲方确认。
- 现场 PLC 点位和 `MODEL_INPUT_COLS` 的映射是否已冻结。
- TabPFN 在目标部署机器上的 GPU/CPU 性能基准。
