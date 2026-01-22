# MSCT-RSMA 代码工程说明

本工程实现论文 **Multi-Satellite Cooperative MIMO Transmission: Statistical CSI-Aware RSMA Precoding Design (WCL 2025)** 的完整仿真流水线，用于生成 P 扫描与 S 扫描的 MMFR 数据与对比图。流程包括：
几何建模 → sCSI 上界构造 → WMMSE 交替优化 → Monte Carlo 遍历率评估 → 基线对比。

## 代码结构与职责

- `msct_rsma/topology.py`：卫星与用户几何建模、覆盖与关联
- `msct_rsma/channel.py`：Rician 信道统计、sCSI 上界 `Hhat`、采样信道
- `msct_rsma/wmmse.py`：WMMSE 交替优化（支持详细迭代日志）
- `msct_rsma/eval.py`：遍历率估计、公共流分配
- `msct_rsma/config.py`：仿真默认参数与常量
- `scripts/simulate_mmfr.py`：仿真入口（只生成数据）
- `scripts/plot_mmfr.py`：绘图入口（只负责画图）
- `paper_text.txt` / `*.pdf`：论文文本与原文

## 环境要求

- Python 3.9+（推荐 3.10/3.11）
- 依赖：`numpy`, `scipy`, `cvxpy`, `matplotlib`

安装依赖：

```bash
pip install -r requirements.txt
```

## 如何运行

建议在工程根目录下运行，并设置 `PYTHONPATH`：

### 1) 生成数据（P/S 扫描）

```bash
PYTHONPATH="$(pwd)" python scripts/simulate_mmfr.py --vs both --out_dir results
```

只跑指定算法（逗号分隔）：

```bash
PYTHONPATH="$(pwd)" python scripts/simulate_mmfr.py --vs P --algs rsma_scsi,sdma_scsi --out_dir results
```

### 2) 快速验证

```bash
PYTHONPATH="$(pwd)" python scripts/simulate_mmfr.py --vs both --quick --out_dir results_quick
```

只跑指定算法：

```bash
PYTHONPATH="$(pwd)" python scripts/simulate_mmfr.py --vs both --quick --algs rsma_scsi,sdma_scsi --out_dir results_quick
```

### 3) 打印 WMMSE 迭代进度

```bash
PYTHONPATH="$(pwd)" python scripts/simulate_mmfr.py --vs both --quick --out_dir results_quick --verbose
```

### 4) 绘图（读取已有数据）

```bash
PYTHONPATH="$(pwd)" python scripts/plot_mmfr.py --in_dir results --out_dir plots --vs both
```

只画指定算法：

```bash
PYTHONPATH="$(pwd)" python scripts/plot_mmfr.py --in_dir results --out_dir plots --vs both --algs rsma_scsi,sdma_scsi
```

## 输出文件与含义

`simulate_mmfr.py` 输出按算法拆分的 JSON 文件：

- `results/P_<alg>.json`：P 扫描下某个算法的平均 MMFR
- `results/S_<alg>.json`：S 扫描下某个算法的平均 MMFR
- 若未指定 `--algs`（运行全部算法），额外输出 `P_all.json` / `S_all.json`

每个 JSON 中包含的指标（可能子集）：

- `rsma_scsi` / `sdma_scsi`：协作 + sCSI
- `rsma_icsi` / `sdma_icsi`：协作 + iCSI（单次瞬时信道设计）
- `rsma_dcsi` / `sdma_dcsi`：协作 + dCSI（LoS-only 设计）
- `rsma_scsi_noncoop` / `sdma_scsi_noncoop`：非协作 + sCSI（最近卫星服务）

## 基线实现说明

基线实现统一封装在 `msct_rsma/baselines.py`，并由 `scripts/simulate_mmfr.py` 调用：

- **RSMA 系列**：`optimize_rsma_scsi / optimize_rsma_icsi / optimize_rsma_dcsi`
- **SDMA 系列**：`optimize_sdma_scsi / optimize_sdma_icsi / optimize_sdma_dcsi`
- **非协作 sCSI**：`optimize_noncoop_rsma_scsi / optimize_noncoop_sdma_scsi`

其中 **SDMA** 通过 `use_common=False` 实现（等价于 `q_c=0`）。

`plot_mmfr.py` 会输出：

- `plots/P.png`
- `plots/S.png`

## 关键参数说明

仿真参数集中在 `msct_rsma/config.py`，常用项：

- `num_drops`：拓扑随机落点次数
- `num_mc_eval`：遍历率 MC 采样次数
- `P_dBW_list` / `S_list`：P/S 扫描列表
- 天线规模、载波频率、带宽、噪声温度等

`--quick` 会把 `num_drops` 改为 3、`num_mc_eval` 改为 2000，用于快速验证。

## 常见问题

### 1) `ModuleNotFoundError: No module named 'msct_rsma'`

请在工程根目录执行并设置 `PYTHONPATH`：

```bash
PYTHONPATH="$(pwd)" python scripts/simulate_mmfr.py --vs both --quick --out_dir results_quick
```

### 2) CVXPY 求解器报错

默认使用 `SCS` 求解器，可检查已安装求解器：

```bash
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

必要时更新或重新安装 `cvxpy` 及其依赖。
