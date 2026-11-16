# 航班运行延误分析与风险评分原型

面向航空公司运行控制场景的个人项目：用公开航班运行数据做延误原因分解、机场/时段热点分析和登机口延误风险评分，验证“数据清洗 → 指标分析 → 时间切分建模 → 可复现报告”的完整链路。

> 这是个人分析原型，不是任何航司的生产系统；所有结论只适用于本项目使用的公开样本，不代表任何航司的实际运行数据。

## 1. 与运控业务问题的对应

| JD 关注点 | 本项目对应 |
| --- | --- |
| 运控业务理解 | 航班延误率、取消率、延误原因（晚到飞机/承运人/NAS/天气/安检） |
| 数据分析与可视化 | 承运人、机场、航线、星期×时段多维统计；机场 GIS 热点图 |
| 科技创新攻关 | 时间切分建模、目标编码防泄漏、风险评分与 Top-decile lift |
| 测绘/GIS 背景 | OurAirports 坐标关联与机场空间分布图 |
| 工程能力 | Python/Pandas/Matplotlib/scikit-learn，脚本化、可复现 |

## 2. 数据

- **航班数据**：Kaggle 公开数据集 *Airline Flight Delay Dataset (100K Records)*（作者 `kesinisoujanya`），BTS 派生 2018 年航班样本。实际下载文件为 **1,048,575 行**，覆盖 **2018-01-01 至 2018-02-26**，18 家承运人、337 个机场。
- **机场坐标**：OurAirports 公开数据 `airports.csv`（US 机场，IATA 代码关联）。
- 数据文件不纳入版本库；复现时按第 5 节下载到 `data/raw/`。数据许可与引用以 Kaggle 数据集页面和 OurAirports 页面为准。

数据口径：

- 完成航班 = `CANCELLED=0` 且 `DIVERTED=0` 且 `ARR_DELAY` 非空。
- 延误 = 到达延误 **> 15 分钟**（DOT 常用口径）。
- 延误原因列只对部分延误航班有归因值；原因占比按“有归因的延误分钟”统计，不等同于全部延误航班。

## 3. 主要结果

样本整体：

- 完成航班 1,019,683；取消 25,913（2.47%）；备降 2,577。
- 到达延误 >15 分钟的比例 **17.94%**；平均到达延误 3.52 分钟，中位数 -7 分钟，P90 36 分钟。
- 承运人最高延误率：**B6 26.55%**；机场最高：**ORD 25.66%**；航线最高（≥500 班）：**SAN–PHX 37.16%**。
- 延误原因（按归因延误分钟）：**晚到飞机 39.10%**、承运人 31.28%、NAS 22.76%、天气 6.72%、安检 0.14%。
- 星期×时段热点：**周五 18:00** 延误率约 **26.5%**（已过滤低样本时段）。

模型（时间切分，测试集为 2018-02-18 至 02-26）：

| 模型 | 信息时点 | ROC-AUC | PR-AUC | Brier | Top 10% 实际延误率 | Lift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 梯度提升（pre-flight） | 起飞前，仅计划时刻+历史统计 | 0.580 | 0.248 | 0.173 | 25.94% | 1.25× |
| 梯度提升（at-gate） | 起飞延误已知（pushback 后） | **0.925** | **0.870** | 0.062 | **99.75%** | **4.80×** |

基线：测试集整体延误率 20.78%。

**结论**：仅靠计划时刻与历史统计，提前预测能力有限（AUC 0.58）；起飞延误是到达延误的主导信号，登机口风险评分可以把 Top 10% 航班的实际延误率从 20.78% 提升到 99.75%。真实运控系统要提升“提前预测”，需要接入天气、流控/NAS、飞机排班与机组等数据——这正是本项目的明确补强方向。

## 4. 目录

```text
csair-flight-ops/
├─ scripts/
│  ├─ 01_prepare.py       # 清洗、派生字段、parquet
│  ├─ 02_descriptive.py   # KPI、承运人/机场/航线、原因、GIS、热力图
│  └─ 03_model.py         # 时间切分、目标编码、pre-flight/at-gate 模型
├─ data/raw/              # 原始数据（不纳入版本库）
├─ data/processed/        # flights.parquet
├─ outputs/figures/       # 图表
├─ outputs/tables/        # 指标表、预测样本、特征重要性
├─ outputs/metrics_*.json # 机器可读指标
├─ report/report.md       # 分析报告
├─ environment.yml
└─ requirements.txt
```

## 5. 复现

本机使用 conda 环境 `csair-ops`（Python 3.11）。依赖已通过 pip 安装：

```bash
conda activate csair-ops
python scripts/01_prepare.py
python scripts/02_descriptive.py
python scripts/03_model.py
```

数据准备：

1. 将 Kaggle 数据集下载为 `data/raw/kaggle_delay/flights.csv`：
   `https://www.kaggle.com/api/v1/datasets/download/kesinisoujanya/airline-flight-delay-dataset-100k-records`
2. 将 OurAirports 数据下载为 `data/raw/airports.csv`：
   `https://davidmegginson.github.io/ourairports-data/airports.csv`

环境文件：

```bash
conda env create -f environment.yml
# 或
pip install -r requirements.txt
```

## 6. 局限

- 样本只覆盖 2018-01-01 至 2018-02-26，且为公开样本，不能代表全年或航司实际运行。
- 未接入天气、NAS/流控、飞机尾号与机组排班数据；pre-flight 模型因此偏弱。
- at-gate 模型使用 `DEP_DELAY`，属于 pushback 后信息，不是“提前预测”；它更适合做**恢复优先级/风险评分**。
- 延误原因列只对部分延误航班有归因，原因占比是“有归因延误分钟”的份额。
- 机场地图使用经纬度散点，没有叠加行政边界；用于热点识别足够，不用于精确制图。
