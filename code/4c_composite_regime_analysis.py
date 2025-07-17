import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from arch import arch_model
from scipy.stats import ttest_ind
from tqdm import tqdm

def analyze_composite_regimes(base_path, factors_path):
    """
    【核心分析脚本】构建四个综合宏观情景指标，并分析因子在其中的表现。
    复现原文Exhibit 5和Exhibit 6的逻辑。
    """
    print("### 开始第四阶段-步骤C: 构建综合宏观情景并分析 ###")

    # 1. 加载所有需要的数据
    print("加载因子、宏观冲击、市场收益和工业增加值数据...")
    path_factors = os.path.join(factors_path, 'all_factors_monthly.csv')
    path_shocks = os.path.join(base_path, 'macro_shocks.csv')
    path_ip_growth = os.path.join(base_path, '月度工业增加值/CME_Mindust1.csv')
    path_index = os.path.join(base_path, '指数文件/TRD_Index.csv') # 用于市场收益
    path_rf = os.path.join(base_path, 'macro_variables_part1.csv') # 用于无风险利率

    df_factors = pd.read_csv(path_factors)
    df_factors['date'] = pd.to_datetime(df_factors['date']) + pd.offsets.MonthEnd(0)

    df_shocks = pd.read_csv(path_shocks)
    df_shocks['date'] = pd.to_datetime(df_shocks['date']) + pd.offsets.MonthEnd(0)

    # 准备工业增加值数据
    df_ip = pd.read_csv(path_ip_growth)
    df_ip = df_ip[(df_ip['Datatype'] == 2) & (df_ip['Datasgn'] == 'A') & (df_ip['Fresgn'] == 'M')]
    df_ip = df_ip[['Staper', 'Eindm0101']]
    df_ip.rename(columns={'Staper': 'date', 'Eindm0101': 'ip_growth'}, inplace=True)
    df_ip['date'] = pd.to_datetime(df_ip['date']) + pd.offsets.MonthEnd(0)
    df_ip['ip_growth'] = df_ip['ip_growth'] / 100 # 转为小数

    # 准备市场超额收益数据
    df_rf = pd.read_csv(path_rf)[['date', 'short_rate']]
    df_rf['date'] = pd.to_datetime(df_rf['date']) + pd.offsets.MonthEnd(0)
    
    df_mkt = pd.read_csv(path_index)
    df_mkt = df_mkt[df_mkt['Indexcd'] == 300]
    df_mkt['date'] = pd.to_datetime(df_mkt['Trddt'])
    mkt_ret = (1 + df_mkt.set_index('date')['Retindex']).resample('ME').prod() - 1
    mkt_ret.name = 'mkt_ret'

    # 合并所有数据到一个主DataFrame
    df = pd.merge(df_factors, df_shocks, on='date')
    df = pd.merge(df, df_ip, on='date')
    df = pd.merge(df, pd.DataFrame(mkt_ret), on='date')
    df = pd.merge(df, df_rf, on='date')
    df['mkt_excess_ret'] = df['mkt_ret'] - df['short_rate']
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # --- 2. 构建四个综合情景指标 ---
    print("正在构建四个综合情景指标...")
    shock_cols = [col for col in df_shocks.columns if '_shock_ortho' in col]
    X = df[shock_cols]

    # 情景1: 风险容忍度 (Risk Tolerance)
    y_rt = df['mkt_excess_ret'].shift(-12) # 未来12个月的市场超额收益
    df_rt = pd.concat([y_rt, X], axis=1).dropna()
    reg_rt = LinearRegression().fit(df_rt[shock_cols], df_rt['mkt_excess_ret'])
    df['risk_tolerance'] = reg_rt.predict(X)

    # 情景2: 宏观展望 (Macro Outlook)
    y_mo = df['ip_growth'].shift(-12) # 未来12个月的工业增加值增长
    df_mo = pd.concat([y_mo, X], axis=1).dropna()
    reg_mo = LinearRegression().fit(df_mo[shock_cols], df_mo['ip_growth'])
    df['macro_outlook'] = reg_mo.predict(X)

    # 情景3: 宏观稳定性 (Macro Stability)
    shocks_for_garch = [col for col in shock_cols if 'market_vol' not in col]
    cond_vols = pd.DataFrame(index=X.index)
    for shock in tqdm(shocks_for_garch, desc="GARCH模型拟合进度"):
        garch_model = arch_model(X[shock], vol='Garch', p=1, q=1)
        res = garch_model.fit(disp='off')
        cond_vols[shock] = res.conditional_volatility
    uncertainty = PCA(n_components=1).fit_transform(cond_vols)
    df['macro_stability'] = -uncertainty # 稳定是“不确定性”的负向

    # 情景4: 风险偏好开关 (Risk-On Conditions)
    df['risk_on_cond'] = 'Mixed'
    df.loc[(df['market_ep_shock_ortho'] < 0) & (df['market_vol_shock_ortho'] < 0), 'risk_on_cond'] = 'Good' # Risk-On
    df.loc[(df['market_ep_shock_ortho'] > 0) & (df['market_vol_shock_ortho'] > 0), 'risk_on_cond'] = 'Bad' # Risk-Off
    print("所有情景指标构建完成。")

    # --- 3. 分析因子在情景下的表现 ---
    print("正在计算因子在各情景下的表现利差...")
    factor_cols = ['SMB', 'HML', 'UMD', 'RMW', 'CMA']
    regime_cols = ['risk_tolerance', 'macro_outlook', 'macro_stability']
    results = []

    # 处理需要分位数的三个情景
    for regime in regime_cols:
        q1 = df[regime].quantile(0.25)
        q4 = df[regime].quantile(0.75)
        good_times = df[df[regime] >= q4]
        bad_times = df[df[regime] <= q1]
        
        for factor in factor_cols:
            ret_good = good_times[factor].mean() * 12
            ret_bad = bad_times[factor].mean() * 12
            spread = ret_good - ret_bad
            _, p_val = ttest_ind(good_times[factor], bad_times[factor], equal_var=False, nan_policy='omit')
            results.append({'Regime': regime, 'Factor': factor, 'Regime Spread (Annualized)': spread, 'P-value': p_val})

    # 处理风险偏好开关情景
    good_times_ro = df[df['risk_on_cond'] == 'Good']
    bad_times_ro = df[df['risk_on_cond'] == 'Bad']
    for factor in factor_cols:
        ret_good = good_times_ro[factor].mean() * 12
        ret_bad = bad_times_ro[factor].mean() * 12
        spread = ret_good - ret_bad
        _, p_val = ttest_ind(good_times_ro[factor], bad_times_ro[factor], equal_var=False, nan_policy='omit')
        results.append({'Regime': 'risk_on_conditions', 'Factor': factor, 'Regime Spread (Annualized)': spread, 'P-value': p_val})
        
    # --- 4. 整理并保存结果 ---
    df_results = pd.DataFrame(results)
    df_pivot = df_results.pivot_table(index='Regime', columns='Factor', values='Regime Spread (Annualized)')
    df_p_values = df_results.pivot_table(index='Regime', columns='Factor', values='P-value')

    def add_stars(val, p_val):
        if p_val < 0.01: return f"{val:.4f}***"
        elif p_val < 0.05: return f"{val:.4f}**"
        elif p_val < 0.1: return f"{val:.4f}*"
        else: return f"{val:.4f}"

    df_display = df_pivot.copy()
    for factor in df_display.columns:
        for regime in df_display.index:
            p_val = df_p_values.loc[regime, factor]
            val = df_pivot.loc[regime, factor]
            df_display.loc[regime, factor] = add_stars(val, p_val)

    final_factor_order = ['SMB', 'HML', 'UMD', 'RMW', 'CMA']
    df_display = df_display[[col for col in final_factor_order if col in df_display.columns]]

    output_path = os.path.join(base_path, 'exhibit6_replication_regime_spreads.csv')
    df_display.to_csv(output_path)
    
    print(f"\n### 综合宏观情景分析完成！ ###")
    print(f"结果已保存至: {output_path}")
    print("\nA股因子在不同宏观情景下的表现 (情景利差):")
    print(df_display)
    print("\n注: *, **, *** 分别代表在10%, 5%, 1%的水平上显著。")


if __name__ == '__main__':
    macro_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/mae'
    factors_base_path = r'/home/shimmer/projects/20250716_factor_investing/data/factor'
    analyze_composite_regimes(macro_base_path, factors_base_path)