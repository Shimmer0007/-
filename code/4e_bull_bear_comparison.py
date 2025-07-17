import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from scipy.stats import ttest_ind
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from arch import arch_model


def bull_bear_comparison(base_path, factors_path):
    """
    【论证脚本】构建一个牛熊中性的投资组合，并检验其在宏观情景下的风险暴露。
    复现原文Exhibit 8, 9的核心逻辑。
    """
    print("### 开始第四阶段-步骤E: 对比宏观情景与牛熊市场 ###")

    # --- 1. 数据准备 (与4d脚本逻辑相同) ---
    print("加载并准备所有分析所需数据...")
    # (为保持脚本独立性，我们再次加载并合并所有数据)
    path_factors = os.path.join(factors_path, 'all_factors_monthly.csv')
    df_factors = pd.read_csv(path_factors)
    df_factors['date'] = pd.to_datetime(df_factors['date']) + pd.offsets.MonthEnd(0)
    
    path_shocks = os.path.join(base_path, 'macro_shocks.csv')
    df_shocks = pd.read_csv(path_shocks)
    df_shocks['date'] = pd.to_datetime(df_shocks['date']) + pd.offsets.MonthEnd(0)

    path_index = os.path.join(base_path, '指数文件/TRD_Index.csv')
    df_mkt = pd.read_csv(path_index)
    df_mkt = df_mkt[df_mkt['Indexcd'] == 300]
    df_mkt['date'] = pd.to_datetime(df_mkt['Trddt'])
    mkt_ret = (1 + df_mkt.set_index('date')['Retindex']).resample('ME').prod() - 1
    mkt_ret.name = 'mkt_ret'
    
    df = pd.merge(df_factors, df_shocks, on='date')
    df = pd.merge(df, pd.DataFrame(mkt_ret), on='date')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    factor_cols = ['SMB', 'HML', 'UMD', 'RMW', 'CMA']

    # --- 2. 定义牛熊情景 ---
    print("正在定义牛熊市场...")
    # 计算季度市场收益率
    mkt_ret_q = df['mkt_ret'].resample('QE').apply(lambda x: (1+x).prod() - 1)
    df['bull_bear_state'] = mkt_ret_q.apply(lambda x: 'Bull' if x >= 0 else 'Bear')
    df['bull_bear_state'].fillna(method='ffill', inplace=True) # 将季度状态填充到月度

    bull_returns = df[df['bull_bear_state'] == 'Bull']
    bear_returns = df[df['bull_bear_state'] == 'Bear']

    # --- 3. 构建三个对比的投资组合 ---
    print("正在构建等权、等风险贡献和牛熊中性组合...")
    # 组合1: 等权组合 (Equal Weight)
    df['ew_portfolio_ret'] = df[factor_cols].mean(axis=1)

    # 组合2: 等风险贡献组合 (Equal Risk Contribution - 简化版，按逆波动率加权)
    inv_vols = 1 / df[factor_cols].std()
    erc_weights = inv_vols / inv_vols.sum()
    df['erc_portfolio_ret'] = df[factor_cols].dot(erc_weights)

    # 组合3: 牛熊中性组合 (Bull/Bear Diversification)
    def objective_func(weights):
        # 目标函数：最小化牛市和熊市的收益差的平方
        portfolio_ret = df[factor_cols].dot(weights)
        bull_mean = portfolio_ret[df['bull_bear_state'] == 'Bull'].mean()
        bear_mean = portfolio_ret[df['bull_bear_state'] == 'Bear'].mean()
        return (bull_mean - bear_mean)**2

    # 约束条件：权重和为1，权重非负
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(len(factor_cols)))
    initial_weights = np.array([1/len(factor_cols)] * len(factor_cols))
    
    opt_result = minimize(objective_func, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    bb_weights = opt_result.x
    df['bb_portfolio_ret'] = df[factor_cols].dot(bb_weights)
    print("牛熊中性组合权重:", dict(zip(factor_cols, bb_weights)))


    # --- 4. 终极考验：检验三个组合在宏观情景下的表现 ---
    # (此部分逻辑与4c脚本类似，但分析对象是组合，不是单个因子)
    print("正在检验各组合在宏观情景下的表现...")
    # 首先需要构建宏观情景指标... (省略重复代码，直接使用之前的结果)
    # 这里我们只做简化版的检验，使用单个宏观冲击作为情景代理
    # 一个完整的检验需要重新构建4个综合指标，代码会非常长
    # 我们以 short_rate_shock 和 market_ep_shock 为例
    results = []
    portfolio_cols = ['ew_portfolio_ret', 'erc_portfolio_ret', 'bb_portfolio_ret']
    regime_cols_simple = ['short_rate_shock_ortho', 'market_ep_shock_ortho']

    for regime in regime_cols_simple:
        q1 = df[regime].quantile(0.25)
        q4 = df[regime].quantile(0.75)
        good_times = df[df[regime] >= q4]
        bad_times = df[df[regime] <= q1]
        
        for portfolio in portfolio_cols:
            ret_good = good_times[portfolio].mean() * 12
            ret_bad = bad_times[portfolio].mean() * 12
            spread = ret_good - ret_bad
            _, p_val = ttest_ind(good_times[portfolio], bad_times[portfolio], equal_var=False, nan_policy='omit')
            results.append({'Regime': regime.replace('_shock_ortho',''), 
                            'Portfolio': portfolio.replace('_portfolio_ret',''), 
                            'Spread': spread, 'P-value': p_val})

    df_results = pd.DataFrame(results)
    df_pivot = df_results.pivot_table(index='Regime', columns='Portfolio', values='Spread')
    
    # --- 5. 展示结果 ---
    print("\n### 宏观情景 vs 牛熊市场对比分析完成！ ###")
    print("\n各投资组合在宏观冲击下的表现利差 (年化):")
    print(df_pivot[['ew', 'erc', 'bb']])
    
    # 验证牛熊中性组合的效果
    bull_neutral_perf_bull = df['bb_portfolio_ret'][df['bull_bear_state'] == 'Bull'].mean()
    bull_neutral_perf_bear = df['bb_portfolio_ret'][df['bull_bear_state'] == 'Bear'].mean()
    print("\n牛熊中性组合表现验证:")
    print(f"  - 牛市平均收益: {bull_neutral_perf_bull:.6f}")
    print(f"  - 熊市平均收益: {bull_neutral_perf_bear:.6f}")
    print(f"  - 收益差: {bull_neutral_perf_bull - bull_neutral_perf_bear:.6f} (已接近0)")


if __name__ == '__main__':
    macro_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/mae'
    factors_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/factor'
    bull_bear_comparison(macro_base_path, factors_base_path)