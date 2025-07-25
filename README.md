# 因子投资宏观风险暴露研究

**(对 "Macroeconomic Risks in Equity Factor Investing" 的本土化复现与探索)**

---

## 1. 项目简介 (Project Overview)

本项目旨在系统性地复现并验证经典文献《Macroeconomic Risks in Equity Factor Investing》的核心研究框架与结论。与原文不同的一点是，我们将目光聚焦于**中国A股市场**，并对宏观经济因素和风险因子的构建进行了全面的本土化适配与实现。

项目的核心目标是回答以下问题：

- A股市场的常用投资风格因子（如小市值、价值、动量、盈利、投资）是否存在显著的宏观风险暴露？
- 当宏观经济环境发生预期之外的“冲击”时，这些因子的表现会如何系统性地偏移？
- 如何将单一的宏观冲击聚合成更稳健的“宏观情景”，并在此框架下理解A股的因子轮动？
- 这些发现对A股市场的多因子组合构建和风险管理有何实践意义？

## 2. 研究流程 (Research Workflow)

整个过程遵循“数据工程 -> 模型复现 -> 分析洞察”的路径，主要分为以下几个阶段：

### 第一阶段：数据工程与因子构建 (Data Engineering & Factor Construction)

这是本项目最核心的基石，也是最痛苦的实践。无论如何，结果确保了所有分析都建立在高质量、高透明度的数据之上。

- **因子构建**: 从最底层的日度/月度行情和财务数据出发，构建A股市场的五个核心风格因子：
  - **SMB**: 市值因子 (Small Minus Big)
  - **HML**: 价值因子 (High Minus Low)
  - **UMD**: 动量因子 (Up Minus Down)
  - **RMW**: 盈利因子 (Robust Minus Weak)
  - **CMA**: 投资因子 (Conservative Minus Aggressive)
- **因子交叉验证**: 将自行构建的因子与CSMAR标准五因子数据库进行交叉验证，通过分析相关系数（SMB>0.96, HML>0.86, RMW>0.78, CMA>0.87），精准定位并修正了CMA因子的构建逻辑，最终确保了所有自建因子的可靠性。
- **宏观指标构建**: 基于本土化考量，收集并处理了6个核心宏观经济指标。其中，**市场整体估值(E/P)**、**市场系统性波动率**和**市场整体流动性(Amihud)** 三个关键指标，是通过对全市场日度交易数据的底层计算与聚合而自行构建的，保证了研究的严谨性。

### 第二阶段：核心分析与模型复现 (Core Analysis & Model Replication)

在数据准备就绪后，严格遵循原论文的计量方法进行分析。

- **宏观意外提取**: 使用**向量自回归模型(VAR(1))** 对6个宏观指标进行建模，提取其残差作为“宏观意外”。随后，通过对市场超额收益进行**正交化**，剥离了市场Beta的影响，得到了纯粹的宏观风险冲击。
- **单一宏观风险分析**: 复现原文`Exhibit 3`的逻辑，通过计算“宏观利差”，量化了A股五大因子对于单一宏观意外（如利率冲击、信用利差冲击等）的敏感度。
- **综合宏观情景分析**: 复现原文`Exhibit 6`的逻辑，将单一宏观冲击聚合成**风险容忍度**、**宏观展望**、**宏观稳定性**、**风险偏好开关**这四个更稳健的宏观情景，并分析了因子在不同情景下的系统性表现。

## 3. 核心发现 (Key Findings)

本研究不仅验证了原论文的核心思想在A股市场同样适用，更发掘了一系列具有本土特色的重要结论：

- **宏观风险普遍存在**: A股因子同样存在显著的宏观风险暴露，其收益在不同宏观环境下并非稳健。
- **SMB因子的独特性**: 与美股不同，A股的小市值因子(SMB)在由基本面驱动的“宏观展望”和“风险容忍度”改善时期，表现出显著的**逆周期性**，这可能与A股市场在牛市中由“龙头白马”驱动的结构性特征有关。
- **双重驱动模式**: A股的因子轮动呈现出“基本面”和“市场情绪”双重驱动的特征。在**基本面改善**时，价值(HML)和盈利(RMW)因子表现更优；而在**市场情绪高涨**（Risk-On）时，高风险的动量(UMD)和小市值(SMB)因子则遥遥领先。
- **向质量/价值飞奔**: 在市场风险溢价走高或流动性收紧时，盈利(RMW)和价值(HML)因子均表现出显著的防御性，印证了“flight-to-quality”的经典逻辑。
- **投资与动量因子的利率敏感性**: 投资(CMA)和动量(UMD)因子均对利率和信用利差的意外冲击呈现显著的负相关性，是管理利率风险时需要重点关注的风险点。

## 4. 结论与启示 (Conclusion & Implications)

分析结果有力地证明，**A股市场的因子投资组合管理必须是“宏观风险认知”下的管理**。一个看似分散的多因子策略，若其成分因子对某种宏观风险（如利率风险）存在同向的负暴露，则在特定宏观环境下可能遭遇集中的、超出预期的回撤。因此，理解并管理因子背后的宏观风险敞口，是构建稳健、高效的多因子投资组合的必要前提。

## 5. 文件结构 (File Structure)

```sh
.
├── code/
│   ├── 1_prepare_data.py               # 整合基础数据
│   ├── 2b_all_factors.py               # 构建全部5个因子
│   ├── 2c*_validate_factors.py          # 交叉验证
│   ├── 3a_process_rates.py             # 处理宏观指标1-2
│   ├── 3b_process_credit_spread.py     # 处理宏观指标3
│   ├── 3c_build_market_indicators.py   # 构建宏观指标4-6
│   ├── 4a_extract_macro_shocks.py      # 提取宏观意外
│   └── 4b_single_macro_risk.py         # 分析单一宏观风险
│   └── 4c_composite_regime_analysis.py # 分析综合宏观情景
│
└── data/
├── factor/
│   ├── all_factors_monthly.csv     # (输出) 最终使用的5因子收益序列
│   └── ...                         # 因子构建相关的原始数据
└── mae/
	├── all_macro_variables.csv     # (输出) 最终的6个宏观指标
	├── macro_shocks.csv            # (输出) 最终的6个宏观意外序列
	├── exhibit3_replication...csv  # (输出) 单一风险分析结果
	├── exhibit6_replication...csv  # (输出) 综合情景分析结果
	└── ...                         # 宏观数据相关的原始数据
```

## 6. 如何运行 (How to Run)

请确保已安装 `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `arch`, `tqdm` 等必要的Python库。 然后按照`code/`文件夹下脚本的**数字前缀顺序** (1 -> 2 -> 3 -> 4) 、调整文件目录依次运行即可完整复现整个研究流程。
